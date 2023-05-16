import os
from os.path import join as opj
import random
import time
import torch
import torch.nn as nn
import numpy as np

from transformers import AdamW, BertConfig
from colbert.utils.runs import Run
from colbert.utils.amp import MixedPrecisionManager
from colbert.utils.utils import save_checkpoint, load_checkpoint
from colbert.training.lazy_batcher import LazyBatcher
from colbert.training.eager_batcher import EagerBatcher
from colbert.parameters import DEVICE
from colbert.training.utils import AverageMeter
from colbert.modeling.colbert import ColBERT
from colbert.modeling.prefix import PrefixColBERT
from colbert.utils.utils import print_message
from colbert.training.utils import print_progress, manage_checkpoints


def train(epoch_idx, args):
    random.seed(12345)
    np.random.seed(12345)
    torch.manual_seed(12345)
    if args.distributed:
        torch.cuda.manual_seed_all(12345)

    if args.distributed:
        assert args.bsize % args.nranks == 0, (args.bsize, args.nranks)
        assert args.accumsteps == 1
        args.bsize = args.bsize // args.nranks

        print("Using args.bsize =", args.bsize, "(per process) and args.accumsteps =", args.accumsteps)

    if args.lazy:
        reader = LazyBatcher(args, (0 if args.rank == -1 else args.rank), args.nranks)
    else:
        reader = EagerBatcher(args, (0 if args.rank == -1 else args.rank), args.nranks)

    if args.rank not in [-1, 0]:
        torch.distributed.barrier()

    if args.prefix:
        config = BertConfig.from_pretrained('bert-base-uncased', cache_dir=".cache")
        config.pre_seq_len = args.pre_seq_len
        config.prefix_hidden_size = args.prefix_hidden_size
        config.prefix_mlp = args.prefix_mlp
        colbert = PrefixColBERT.from_pretrained('bert-base-uncased', config=config,
                                          query_maxlen=args.query_maxlen,
                                          doc_maxlen=args.doc_maxlen,
                                          dim=args.dim,
                                          similarity_metric=args.similarity,
                                          mask_punctuation=args.mask_punctuation)
    else:
        config = BertConfig.from_pretrained('bert-base-uncased', cache_dir=".cache")
        colbert = ColBERT.from_pretrained('bert-base-uncased',
                                        config=config,
                                        query_maxlen=args.query_maxlen,
                                        doc_maxlen=args.doc_maxlen,
                                        dim=args.dim,
                                        similarity_metric=args.similarity,
                                        mask_punctuation=args.mask_punctuation)

    last_checkpoint_path = opj(opj(Run.path, 'checkpoints'), "colbert-epoch{}-{}.dnn".format(epoch_idx-1, -1))
    if os.path.exists(last_checkpoint_path):
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        last_checkpoint = torch.load(last_checkpoint_path, map_location='cpu')   # last checkpoint including prefix

        checkpoint['model_state_dict'].update(last_checkpoint['model_state_dict'])
        colbert.load_state_dict(checkpoint['model_state_dict'], strict=False)   # bert pos id can miss
    elif args.checkpoint is not None:
        # assert args.resume_optimizer is False, "TODO: This would mean reload optimizer too."
        print_message(f"#> Starting from checkpoint {args.checkpoint} -- but NOT the optimizer!")

        checkpoint = torch.load(args.checkpoint, map_location='cpu')

        try:
            colbert.load_state_dict(checkpoint['model_state_dict'])
        except:
            print_message("[WARNING] Loading checkpoint with strict=False")
            colbert.load_state_dict(checkpoint['model_state_dict'], strict=False)

    if args.rank == 0:
        torch.distributed.barrier()

    colbert = colbert.to(DEVICE)
    colbert.train()

    if args.distributed:
        colbert = torch.nn.parallel.DistributedDataParallel(colbert, device_ids=[args.rank],
                                                            output_device=args.rank,
                                                            find_unused_parameters=True)

    optimizer = AdamW(filter(lambda p: p.requires_grad, colbert.parameters()), lr=args.lr, eps=1e-8)
    optimizer.zero_grad()

    amp = MixedPrecisionManager(args.amp)
    criterion = nn.CrossEntropyLoss()
    labels = torch.zeros(args.bsize, dtype=torch.long, device=DEVICE)

    start_time = time.time()
    train_loss = 0.0

    start_batch_idx = 0

    if args.resume:
        assert args.checkpoint is not None
        start_batch_idx = checkpoint['batch']
        reader.skip_to_batch(start_batch_idx, checkpoint['arguments']['bsize'])
        if args.resume_optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if os.path.exists(last_checkpoint_path):
        # load optimizer for multiple epochs finetuning
        optimizer.load_state_dict(last_checkpoint['optimizer_state_dict'])

    for batch_idx, BatchSteps in zip(range(start_batch_idx, args.maxsteps), reader):
        this_batch_loss = 0.0

        positive_score_meter = AverageMeter('Positive Score', ':.6f')
        negative_score_meter = AverageMeter('Negative Score', ':.6f')
        score_meter_group = {'pos': positive_score_meter, 'neg': negative_score_meter}
        # (q_ids, q_masks), CLS, [Q], q_content, SEP; (d_ids, d_masks), CLS, [D], d_content, SEP.
        for queries, passages in BatchSteps:
            with amp.context():
                scores = colbert(queries, passages)     # 2bs *
                scores = scores.view(2, -1).permute(1, 0)
                loss = criterion(scores, labels[:scores.size(0)])   # triplet loss (positive pairs should be closer than negatives
                loss = loss / args.accumsteps

            if args.rank < 1 and batch_idx % 100 == 0:
                pos_score_avg, neg_score_avg = print_progress(scores, len(queries[0]), score_meter_group)
                print("#Score>>>   ", pos_score_avg, neg_score_avg, '\t\t|\t\t', pos_score_avg - neg_score_avg)
            amp.backward(loss)

            train_loss += loss.item()
            this_batch_loss += loss.item()

        amp.step(colbert, optimizer)

        if args.rank < 1:
            avg_loss = train_loss / (batch_idx+1)

            num_examples_seen = (batch_idx - start_batch_idx) * args.bsize * args.nranks
            elapsed = float(time.time() - start_time)

            log_to_mlflow = (batch_idx % 20 == 0)
            Run.log_metric('train/avg_loss', avg_loss, step=batch_idx, log_to_mlflow=log_to_mlflow)
            Run.log_metric('train/batch_loss', this_batch_loss, step=batch_idx, log_to_mlflow=log_to_mlflow)
            Run.log_metric('train/examples', num_examples_seen, step=batch_idx, log_to_mlflow=log_to_mlflow)
            Run.log_metric('train/throughput', num_examples_seen / elapsed, step=batch_idx, log_to_mlflow=log_to_mlflow)

            if batch_idx % 100 == 0:
                print_message(batch_idx, avg_loss)
            manage_checkpoints(args, colbert, optimizer, batch_idx+1, epoch_idx)


    name = opj(opj(Run.path, 'checkpoints'), "colbert-epoch{}-{}.dnn".format(epoch_idx, -1))
    save_checkpoint(name, epoch_idx, -1, colbert, optimizer, args.input_arguments.__dict__, args.prefix)
