import os
import torch
from test_config import args
from metrics import accuracy, compute_metric
import json
from utils import AverageMeter, load_documents, load_candidates
from torch.utils.data import DataLoader
import logging

logging.basicConfig(level=logging.INFO)


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


def evaluate(test_loader, model, args):

    accs = AverageMeter('Acc', ':6.3f')
    mean_rank = AverageMeter('mean rank', ':6.3f')
    hit1 = AverageMeter('hit1', ':6.3f')
    hit3 = AverageMeter('hit3', ':6.3f')
    hit10 = AverageMeter('hit3', ':6.3f')
    mrr = AverageMeter('MRR', ':6.3f')
    model.eval()

    total_length = len(test_loader)
    logging.warning("Evaluating with brute force search to reproduce results reported in the paper. There is CONSIDERABLE waiting time.")
    for i, batch_data in enumerate(test_loader):
        if torch.cuda.is_available():
            batch_data = move_to_cuda(batch_data)

        labels = batch_data["labels"]  # bs x 1
        batch_size = len(labels)

        with torch.no_grad():
            if args.model_type == 'bi_encoder':
                mention_dicts = batch_data["mention_dicts"]  # bs x 768
                candidate_dicts = batch_data["candidate_dicts"]
                logits, metrics = model.predict(mention_dicts, candidate_dicts, labels)
            elif args.model_type == 'cross_encoder':
                del batch_data["labels"]
                logits = model(**batch_data)
            else:
                raise NotImplementedError

            metrics = compute_metric(logits, labels)
            predictions = logits.argmax(1)
            _acc = torch.sum(torch.eq(predictions, labels)) / len(labels)

        accs.update(_acc.item(), batch_size)
        mean_rank.update(metrics['mean_rank'])
        mrr.update(metrics['mrr'], batch_size)
        hit1.update(metrics['hit1'], batch_size)
        hit3.update(metrics['hit3'], batch_size)
        hit10.update(metrics['hit10'], batch_size)

        logging.info("Processing iteration {}/{}: MRR:{:.3f}, Acc: {:.3f}, @Hit1 {:.3f}, @Hit3 {:.3f}, @Hit10 {:.3f}".format(i, total_length, mrr.avg, accs.avg, hit1.avg, hit3.avg, hit10.avg))
    metrics = {"mean_rank": mean_rank.avg, "mrr": mrr.avg, "hit1": hit1.avg, "hit3": hit3.avg, "hit10": hit10.avg}
    with open(os.path.join('/'.join(args.eval_model_path.split('/')[:-1]), "{}_test_metric.json".format(args.task)), 'w', encoding='utf-8') as f:
        f.write(json.dumps(metrics, indent=4))


def main():
    if args.task == 'cross_domain':
        assert ('da' in args.test_mentions_file), 'domain adaptation keyword must exist in test mention path'
    all_documents = {}      # doc_id/ entity_id to entity
    document_path = args.document_files[0].split(",")
    for input_file_path in document_path:
        all_documents.update(load_documents(input_file_path))

    if args.model_type == 'bi_encoder':
        from preprocessing.bi_preprocess_data import EntityLinkingSet, compose_collate
        from modeling.bi_encoders import EntityLinkingPredictor

        test_dataset = EntityLinkingSet(
            document_files=all_documents,
            mentions_path=args.test_mentions_file,
            tfidf_candidates_file=args.test_tfidf_candidates_file,
            num_candidates=args.num_candidates,
            max_seq_length=args.max_seq_length,
            is_training=False)
        test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False, num_workers=8, collate_fn=compose_collate, pin_memory=True)
        model = EntityLinkingPredictor(args.eval_model_path).cuda()

    elif args.model_type == 'cross_encoder':
        from preprocessing.preprocess_data import EntityLinkingSet, collate
        from modeling.cross_encoder import EntityLinkingPredictor

        test_dataset = EntityLinkingSet(
            document_files=all_documents,
            mentions_path=args.test_mentions_file,
            tfidf_candidates_file=args.test_tfidf_candidates_file,
            num_candidates=args.num_candidates,
            max_seq_length=args.max_seq_length,
            is_training=False)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=8, collate_fn=collate, pin_memory=True)
        model = EntityLinkingPredictor(args.eval_model_path).cuda()
    else:
        raise NotImplementedError

    evaluate(test_loader, model, args)


if __name__ == "__main__":
    main()

