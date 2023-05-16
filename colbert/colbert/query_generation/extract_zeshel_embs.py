from colbert.modeling.inference import ModelInference
import json
from collections import defaultdict
from colbert.evaluation.loaders import load_colbert
from colbert.utils.parser import Arguments
import random
import numpy as np
import torch
import torch.nn.functional as F
import os


def prepare_encoders(args):
    colbert, _ = load_colbert(args)
    colbert.cuda()
    colbert.eval()
    inference = ModelInference(colbert, amp=True)
    return inference


def process_zeshel_sentences(sample_list):
    domain2sents = defaultdict(list)
    for sample_dict in sample_list:
        mid, mention, context, corpus, entity, entity_desc = sample_dict["mention_id"], sample_dict["mention"], \
        sample_dict[
            "context"], sample_dict["corpus"], sample_dict["entity"], sample_dict["entity_desc"]
        sent = "{}: {}. {}: {}".format(mention, context, entity, entity_desc)
        domain2sents[corpus].append(sent)
    return domain2sents


def process_scifact_sentences(sample_list):
    sentences = []
    mid2sentence_id = {}
    sentence_id2mid = {}
    for i, sample_dict in enumerate(sample_list):
        mid, mention, context, corpus, entity_desc, mention_type = sample_dict["_id"], sample_dict["mention"], sample_dict["text"],\
            sample_dict["domain"], sample_dict["answer"], sample_dict["mention type"]
        sent = "{}: {}. {}".format(mention, context, entity_desc)
        sentences.append(sent)
        mid2sentence_id[str(mid)] = str(i)
        sentence_id2mid[str(i)] = str(mid)
    return sentences, mid2sentence_id, sentence_id2mid


def extract_zeshel_embs(model, samples, args):
    domain2sents = process_zeshel_sentences(samples)
    bsz = args.bsize
    all_results = []
    for domain, sentences in domain2sents.items():
        n = len(sentences)
        d_results = []
        for j, batch_start in enumerate(range(0, n, bsz)):
            batch_text = sentences[batch_start: batch_start + bsz]
            embs = model.docFromText(batch_text, bsize=32, keep_dims=True)
            embs = torch.mean(embs, dim=1)
            assert len(embs) == len(batch_text)
            embs = embs.cpu().numpy()
            d_results.append(embs)

        d_results = np.concatenate(d_results, axis=0)
        all_results.append(d_results)
        np.save("./colbert/query_generation/zeshel_colbert/{}.npy".format(domain), d_results)
    np.concatenate(all_results, axis=0)
    np.save("./colbert/query_generation/zeshel_colbert/all_claims.npy", all_results)


def extract_scifact_embs(model, samples, args):
    bsz = args.bsize
    n = len(samples)
    d_results = []
    for j, batch_start in enumerate(range(0, n, bsz)):
        batch_text = sentences[batch_start: batch_start + bsz]
        embs = model.docFromText(batch_text, bsize=32, keep_dims=True)
        embs = torch.mean(embs, dim=1)  # mean of tokens from a passage as representation
        embs = F.normalize(embs, p=2, dim=1)
        assert len(embs) == len(batch_text)
        embs = embs.cpu().numpy()
        d_results.append(embs)

    d_results = np.concatenate(d_results, axis=0)
    np.save("./colbert/query_generation/scifact_colbert/{}.npy".format('scifacts'), d_results)


if __name__ == "__main__":

    random.seed(12345)

    parser = Arguments(description='Extract embeddings with ColBERT.')

    parser.add_model_parameters()
    parser.add_tuning_parameters()
    parser.add_model_inference_parameters()
    parser.add_ranking_input()
    parser.add_argument('--dataset-type', required=True, type=str,
                        default="scifact",
                        help='scifact or fandomwiki')
    args = parser.parse()

    model = prepare_encoders(args)
    print("model loaded successfully!")
    with open(args.queries, "r") as f:
        data_samples = json.load(f)

    sample_id2idx = {}

    if args.dataset_type == "scifact":
        sentences, mid2fid, fid2mid = process_scifact_sentences(data_samples)
        with open("../data/scifact/mention_id2fid.json", "w") as f:
            json.dump(mid2fid, f)
        with open("../data/scifact/fid2mention_id.json", "w") as f:
            json.dump(fid2mid, f)
        extract_scifact_embs(model, sentences, args)
    elif args.dataset_type == 'fandomwiki':
        extract_zeshel_embs(model, data_samples, args)


