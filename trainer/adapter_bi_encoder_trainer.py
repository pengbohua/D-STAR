import copy
import torch.nn as nn
import torch.utils.data
import time
import json
import torch
import os
import shutil
import glob
from typing import Dict, List, Optional
from transformers import AdamW, get_linear_schedule_with_warmup
from preprocessing.bi_preprocess_data import compose_collate
from utils import AverageMeter, ProgressMeter, logger
from transformers import AutoConfig
from dataclasses import dataclass, field
from metrics import accuracy, compute_metric
# import wandb

curr_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))

@dataclass
class TrainingArguments:
    learning_rate: float = field(default=1e-5,
                            metadata={"help": "learning rate for optimization"}
                            )
    weight_decay: float = field(default=1e-4,
                            metadata={"help": "weight decay parameter for optimization"}
                            )
    grad_clip: float = field(default=10,
                            metadata={"help": "magnitude for gradient clipping"}
                            )
    warmup: int = field(default=500,
                        metadata={"help": "warmup steps"})
    use_amp: bool = field(default=False,
                        metadata={"help": "use mixed precision"})
    train_batch_size: int = field(default=16,
                        metadata={"help": "train batch size"})
    eval_batch_size: int = field(default=16,
                        metadata={"help": "eval batch size"})
    eval_every_n_intervals: int = field(default=2,
                        metadata={"help": "eval every n steps"})
    log_every_n_intervals: int = field(default=100,
                        metadata={"help": "log every n steps"})
    max_weights_to_keep: int = field(default=3,
                                     metadata={"help": "max number of weight file to keep"})


class Trainer:

    def __init__(self,
                 args,
                 pretrained_model_path,
                 eval_model_path,
                 train_dataset,
                 eval_dataset,
                 learning_rate,
                 training_epochs,
                 train_args,
                 model_type='contrastive',
                 num_candidates=64,
                 num_workers=12,
                 use_tf_idf_negatives=True,
                 use_in_batch_mention_negatives=False,
                 use_rdrop=True,
                 margin=0.00,
                 prefix=False,
                 prefix_length=180,
                 ):
        # training arguments
        self.args = train_args
        self.epochs = training_epochs
        self.num_candidates = num_candidates
        self.pretrained_model_path = pretrained_model_path
        self.eval_model_path = eval_model_path + "/" + curr_time
        os.makedirs(self.eval_model_path, exist_ok=True)

        self.use_tfidf_negatives = True if (self.num_candidates != 0 and use_tf_idf_negatives) else False
        self.use_in_batch_mention_negatives = use_in_batch_mention_negatives

        self.use_rdrop = use_rdrop
        # create model
        logger.info("Creating model")
        self.config = AutoConfig.from_pretrained(pretrained_model_path)

        if model_type == 'contrastive' and args.lora:
            from modeling.adapter_bi_encoders import EntityLinker
            self.model = EntityLinker(lora=True, r=64)
        elif model_type == 'contrastive' and args.bitfit:
            from modeling.adapter_bi_encoders import EntityLinker
            self.model = EntityLinker(bitfit=True)
        elif model_type == 'contrastive' and args.adapter:
            from modeling.adapter_bi_encoders import EntityLinker
            self.model = EntityLinker(adapter=True)
        else:
            raise NotImplementedError
        self.model.additive_margin = margin

        if self.args.use_amp:
            self.model, self.optimizer = amp.initialize(self.model.cuda(), self.optimizer, opt_level="O1")
        self._setup_training()

        self.criterion = nn.CrossEntropyLoss().cuda()
        self.optimizer = AdamW([p for p in self.model.parameters() if p.requires_grad],
                               lr=self.args.learning_rate,
                               eps=1e-8)

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
            collate_fn=compose_collate,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True)

        self.valid_loader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=self.args.eval_batch_size,
            shuffle=False,
            collate_fn=compose_collate,
            num_workers=num_workers,
            pin_memory=True)

    def run(self):

        # wandb.watch(self.model)
        for epoch in range(self.epochs):
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
            elif type(maybe_tensor) == list:
                return [_move_to_cuda(t) for t in maybe_tensor]
            else:
                for key, value in maybe_tensor.items():
                    maybe_tensor[key] = _move_to_cuda(value)
                return maybe_tensor
        return _move_to_cuda(sample)

    @torch.no_grad()
    def evaluate(self, step=0):
        if not self.is_training:
            metric_dict = self.eval_loop()
        else:
            metric_dict = self.eval_loop()

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
                                     is_best=True, filename=os.path.join(self.eval_model_path, "best_model.ckpt"))

            else:
                filename = '{}/checkpoint_{}_{}.ckpt'.format(self.eval_model_path, self.epoch, step)
                self.save_checkpoint(checkpoint_dict, is_best=False, filename=filename)

            self.delete_old_ckt(path_pattern='{}/checkpoint_*.ckpt'.format(self.eval_model_path),
                       keep=self.args.max_weights_to_keep)
        # wandb.log(metric_dict)
        logger.info(metric_dict)

    @torch.no_grad()
    def eval_loop(self) -> Dict:

        losses = AverageMeter('Loss', ':.4')
        accs = AverageMeter('Acc', ':6.2f')
        hit1 = AverageMeter('hit1', ':6.2f')
        hit3 = AverageMeter('hit3', ':6.2f')
        hit10 = AverageMeter('hit10', ':6.2f')
        mrr = AverageMeter('MRR', ':6.2f')

        for i, batch_cl_data in enumerate(self.valid_loader):
            self.model.eval()

            if torch.cuda.is_available():
                batch_cl_data = self.move_to_cuda(batch_cl_data)

            mention_dicts = batch_cl_data["mention_dicts"]  # 1024 x 768
            labels = batch_cl_data["labels"]    # 1024 x 1
            candidate_dicts = batch_cl_data["candidate_dicts"]

            batch_size = len(labels)
            logits, metrics = self.get_model_obj(self.model).predict(mention_dicts, candidate_dicts, labels)
            loss = self.criterion(logits, labels)
            losses.update(loss.item(), batch_size)

            predictions = logits.argmax(1)
            _acc = torch.sum(torch.eq(predictions, labels)) / len(labels)

            accs.update(_acc.item(), batch_size)
            mrr.update(metrics['mrr'], batch_size)
            hit1.update(metrics['hit1'], batch_size)
            hit3.update(metrics['hit3'], batch_size)
            hit10.update(metrics['hit10'], batch_size)

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

        if self.use_rdrop:
            rdrop_losses = AverageMeter('R-drop Loss', ':.4')
        progress = ProgressMeter(
            len(self.train_loader),
            [losses, accs, rdrop_losses] if self.use_rdrop else [losses, accs],
            prefix="Epoch: [{}]".format(self.epoch))
        
        log_every_n_steps = int(len(self.train_loader) / self.args.log_every_n_intervals)
        eval_every_n_steps = int(len(self.train_loader) / (self.args.eval_every_n_intervals + 1))
        for i, batch_cl_data in enumerate(self.train_loader):
            # switch to train mode
            self.model.train()

            if torch.cuda.is_available():
                batch_cl_data = self.move_to_cuda(batch_cl_data)

            mention_dicts = batch_cl_data["mention_dicts"]
            entity_dicts = batch_cl_data["entity_dicts"]
            candidate_dicts = batch_cl_data["candidate_dicts"]
            # avoid contrastive on different mentions to the same entity within the batch
            mm_mask = batch_cl_data["mm_mask"]
            # avoid contrastive on one mention to different entities (different entities sharing the same alias is rare)
            me_mask = batch_cl_data["me_mask"]
            batch_size = len(mention_dicts['input_ids'])

            # compute output
            if not self.use_tfidf_negatives:
                output_dicts = self.model(mention_dicts=mention_dicts, entity_dicts=entity_dicts)
                del candidate_dicts
            else:
                output_dicts = self.model(mention_dicts=mention_dicts, entity_dicts=entity_dicts, candidate_dict_list=candidate_dicts)

            if not self.use_in_batch_mention_negatives:
                mm_mask = None

            logits = self.get_model_obj(self.model).compute_logits(me_mask, mm_mask, **output_dicts)

            labels = torch.arange(len(logits)).to(logits.device)

            predictions = logits.argmax(1)
            _acc = torch.sum(torch.eq(predictions, labels)) / len(labels)

            loss = self.criterion(logits, labels)
            loss += self.criterion(logits[:, :batch_size].t(), labels)

            if self.use_rdrop:
                logits2 = self.get_model_obj(self.model).compute_logits(me_mask, mm_mask, **output_dicts) # dropout for regularization
                loss = self.criterion(logits, labels)
                loss2 = self.criterion(logits2, labels)
                rdrop_loss = self.get_model_obj(self.model).compute_kl_loss(logits, logits2)
                loss = 0.5 * (loss + loss2) + 1e5 * rdrop_loss
                rdrop_losses.update((1e5 * rdrop_loss).item(), batch_size)

            accs.update(_acc.item(), batch_size)
            losses.update(loss.item(), batch_size)

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            if self.args.use_amp:
                # apex 
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
                    torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), self.args.grad_clip)

                self.optimizer.step()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                self.optimizer.step()
            self.scheduler.step()

            # wandb.log({"training loss": loss.item()})
            if i % log_every_n_steps == 0:
                progress.display(i)
            if (i + 1) % eval_every_n_steps == 0:
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

