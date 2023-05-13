import torch.nn as nn
import torch.utils.data
import time
import json
import torch
import os
import shutil
import glob
from typing import Dict, List
from transformers import AdamW, get_linear_schedule_with_warmup
from preprocessing.preprocess_data import collate
from utils import AverageMeter, ProgressMeter, logger
from transformers import BertModel, AutoConfig
from dataclasses import dataclass, field
from metrics import accuracy, compute_metric
from train_config import TrainingArguments


curr_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))


class Trainer:

    def __init__(self,
                 args,
                 pretrained_model_path,
                 eval_model_path,
                 train_dataset,
                 eval_dataset,
                 learning_rate=1e-5,
                 num_workers=4,
                 model_type='cross_encoder',
                 train_args: TrainingArguments = None
                 ):
        # training arguments
        self.args = train_args
        self.num_candidates = self.args.num_cand
        self.pretrained_model_path = pretrained_model_path
        self.eval_model_path = eval_model_path + "/" + curr_time
        os.makedirs(self.eval_model_path, exist_ok=True)
        # create model
        logger.info("Creating model")
        if model_type == 'cross_encoder' and args.lora:
            from modeling.adapter_cross_encoders import CrossEncoder
            self.model = CrossEncoder(lora=True, r=64)
        elif model_type == 'cross_encoder' and args.bitfit:
            from modeling.adapter_cross_encoders import CrossEncoder
            self.model = CrossEncoder(bitfit=True)
        elif model_type == 'cross_encoder' and args.adapter:
            from modeling.adapter_cross_encoders import CrossEncoder
            self.model = CrossEncoder(adapter=True)
        else:
            raise NotImplementedError
        self.model = self.model.cuda()

        # loss and optimization
        self.positive_weight = torch.FloatTensor([8]).cuda()     # neg counts / pos counts
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.positive_weight).cuda()

        self.optimizer = AdamW([p for p in self.model.parameters() if p.requires_grad],
                               lr=learning_rate,
                               weight_decay=self.args.weight_decay)

        if self.args.use_amp:
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level="O1")

        self._setup_training()

        num_training_steps = self.args.epochs * len(train_dataset) // max(self.args.train_batch_size, 1)
        self.args.warmup = min(self.args.warmup, num_training_steps // 10)
        logger.info('Total training steps: {}, warmup steps: {}'.format(num_training_steps, self.args.warmup))
        self.scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer,
                                                   num_warmup_steps=self.args.warmup,
                                                   num_training_steps=num_training_steps)

        # initial status
        self.is_training = True
        self.best_metric = None
        self.epoch = 0
        # dataloader
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size,
            shuffle=True,
            collate_fn=collate,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True)

        self.valid_loader = torch.utils.data.DataLoader(
                eval_dataset,
                batch_size=self.args.eval_batch_size,
                shuffle=False,
                collate_fn=collate,
                num_workers=num_workers,
                pin_memory=True)

    def run(self):

        for epoch in range(self.args.epochs):
            self.train_one_epoch()
            self.evaluate()
            self.epoch += 1

    @staticmethod
    def move_to_cuda(sample):
        if len(sample) == 0:
            return {}

        def _move_to_cuda(maybe_tensor):
            if torch.is_tensor(maybe_tensor):
                return maybe_tensor.cuda(non_blocking=True)
            else:
                for key, value in maybe_tensor.items():
                    maybe_tensor[key] = _move_to_cuda(value)
                return maybe_tensor
        return _move_to_cuda(sample)

    @torch.no_grad()
    def evaluate(self, step=0):
        metric_dict = self.eval_loop()
        if not self.is_training:
            pass    # do not save checkpoint
        else:
            if self.args.use_amp:
                checkpoint_dict = {"state_dict": self.model.state_dict(),
                                   "amp": amp.state_dict(),
                                   "optimizer": self.optimizer.state_dict()
                                   }
            else:
                checkpoint_dict = {"state_dict": self.model.state_dict(),
                                   "optimizer": self.optimizer.state_dict()
                                   }

            if self.best_metric is None or metric_dict['hit1'] > self.best_metric['hit1']:
                self.best_metric = metric_dict
                with open(os.path.join(self.eval_model_path, "best_metric"), 'w', encoding='utf-8') as f:
                    f.write(json.dumps(metric_dict, indent=4))


                self.save_checkpoint(checkpoint_dict,
                                     is_best=True, filename=os.path.join(self.eval_model_path, "model_best.ckpt"))

            else:
                filename = '{}/checkpoint_{}_{}.ckpt'.format(self.eval_model_path, self.epoch, step)
                self.save_checkpoint(checkpoint_dict, is_best=False, filename=filename)

            self.delete_old_ckt(path_pattern='{}/checkpoint_*.ckpt'.format(self.eval_model_path),
                       keep=self.args.max_weights_to_keep)

        logger.info(metric_dict)

    @torch.no_grad()
    def eval_loop(self) -> Dict:

        losses = AverageMeter('Loss', ':.4')
        accs = AverageMeter('Acc', ':6.2f')
        hit1 = AverageMeter('hit1', ':6.2f')
        hit3 = AverageMeter('hit3', ':6.2f')
        hit10 = AverageMeter('hit10', ':6.2f')
        mrr = AverageMeter('MRR', ':6.2f')

        for i, (batch_dict, labels) in enumerate(self.valid_loader):
            self.model.eval()

            if torch.cuda.is_available():
                batch_dict = self.move_to_cuda(batch_dict)
                labels = labels.cuda(non_blocking=True)
            batch_size = len(labels)

            outputs = self.model(**batch_dict)
            h = outputs.last_hidden_state[:, 0, :]
            logits = self.get_model_obj(self.model).predict_head(h) * self.t

            loss = self.criterion(logits.squeeze(), labels.float())
            losses.update(loss.item(), batch_size)

            acc = accuracy(logits, labels)
            metrics = compute_metric(logits, labels)
            accs.update(acc.item(), batch_size//self.num_candidates)

            mrr.update(metrics['mrr'], metrics['chunk_size'])
            hit1.update(metrics['hit1'], metrics['chunk_size'])
            hit3.update(metrics['hit3'], metrics['chunk_size'])
            hit10.update(metrics['hit10'], metrics['chunk_size'])

        metric_dict = {'acc': round(accs.avg, 3),
                       'mrr': round(mrr.avg, 3),
                       'hit1': round(hit1.avg, 3),
                       'hit3': round(hit3.avg, 3),
                       'hit10': round(hit10.avg, 3),
                       'loss': round(losses.avg, 3)}
        logger.info('Epoch {}, valid metric: {}'.format(self.epoch, json.dumps(metric_dict)))
        return metric_dict

    def train_one_epoch(self):
        losses = AverageMeter('Loss', ':.4')
        accs = AverageMeter('Acc', ':6.2f')
        progress = ProgressMeter(
            len(self.train_loader),
            [losses, accs],
            prefix="Epoch: [{}]".format(self.epoch))

        for i, batch_dict in enumerate(self.train_loader):
            self.model.train()

            labels = batch_dict['labels']
            del batch_dict['labels']
            if torch.cuda.is_available():
                batch_dict = self.move_to_cuda(batch_dict)
                labels = labels.cuda()

            batch_size = len(labels)
            logits = self.model(**batch_dict)
            loss = self.criterion(logits.squeeze(), labels.float())

            acc = accuracy(logits, labels)

            accs.update(acc.item(), batch_size // self.num_candidates)
            losses.update(loss.item(), batch_size)

            self.optimizer.zero_grad()
            if self.args.use_amp:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
                    torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), self.args.grad_clip)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
            self.optimizer.step()
            self.scheduler.step()

            if i % self.args.log_every_n_steps == 0:
                progress.display(i)
            if (i + 1) % self.args.eval_every_n_steps == 0:
                self.evaluate(step=i)

    @staticmethod
    def get_model_obj(model: nn.Module):
        return model.module if hasattr(model, "module") else model

    @staticmethod
    def save_checkpoint(state: dict, is_best: bool, filename: str):
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, os.path.dirname(filename) + '/model_best.ckpt')
        shutil.copyfile(filename, os.path.dirname(filename) + '/model_last.ckpt')

    @staticmethod
    def delete_old_ckt(path_pattern: str, keep=5):
        files = sorted(glob.glob(path_pattern), key=os.path.getmtime, reverse=True)     # glob search cur dir with path_pattern
        for f in files[keep:]:
            logger.info('Delete old checkpoint {}'.format(f))
            os.system('rm -f {}'.format(f))

    def _setup_training(self):
        if torch.cuda.device_count() > 1:
            logger.info("Training with {} GPUs in parallel".format(torch.cuda.device_count()))
            self.model = torch.nn.DataParallel(self.model).cuda()
        elif torch.cuda.is_available():
            self.model.cuda()
        else:
            logger.info("Training with CPU")
