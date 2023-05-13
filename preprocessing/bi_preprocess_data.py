from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import tensor
import collections
import json
import os
import math
import random
from transformers import AutoConfig, AutoTokenizer
import json
from dataclasses import dataclass, field
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple
from preprocessing.utils import get_mention_mask, get_label_mask, normalize_context
from argparse import ArgumentParser
import csv
import numpy as np
import threading
import queue


def load_candidates(input_dir):
    documents = {}
    with open(input_dir, "r", encoding="utf-8") as reader:
        doc_dicts = json.load(reader)

    for doc in doc_dicts:
        documents[doc['mention_id']] = doc["tfidf_candidates"]  # mention_id to candidates

    return documents

def load_documents(input_dir):
    '''
    load all documents of Zeshel into a large dictionary
    '''
    documents = {}
    with open(input_dir, "r", encoding="utf-8") as reader:
        domain = input_dir.split('/')[-1].replace('_', ' ').replace('.json', '')
        while True:
            line = reader.readline()
            line = line.strip()
            if not line:
                break
            line = json.loads(line)
            line['metadata'] = domain
            documents[line['document_id']] = line
    print("Loading {} documents from {}".format(len(documents), input_dir))
    return documents


@dataclass
class CLInstance(object):
    """A single set of features for EL as a contrastive learning task"""
    mention_dict: Dict
    entity_dict: Dict
    candidate_dicts: List[Dict]
    mention_id: str
    label_id: str
    label: tensor
    corpus: str


def get_context_tokens(tokenizer, context_tokens, start_index, end_index, max_tokens):
    " extract mention context with an evenly distributed sliding window"
    start_pos = start_index - max_tokens
    if start_pos < 0:
        start_pos = 0
    prefix = ' '.join(context_tokens[start_pos: start_index])
    suffix = ' '.join(context_tokens[end_index + 1: end_index + max_tokens + 1])
    prefix = tokenizer.tokenize(prefix)
    suffix = tokenizer.tokenize(suffix)
    mention = tokenizer.tokenize(' '.join(context_tokens[start_index: end_index + 1]))

    assert len(mention) < max_tokens

    remaining_tokens = max_tokens - len(mention)
    half_remaining_tokens = int(math.ceil(1.0 * remaining_tokens / 2))

    # protect the shorter side
    if len(prefix) >= half_remaining_tokens and len(suffix) >= half_remaining_tokens:
        prefix_len = half_remaining_tokens
    elif len(prefix) >= half_remaining_tokens and len(suffix) < half_remaining_tokens:
        prefix_len = remaining_tokens - len(suffix)
        if prefix_len > len(prefix):
            prefix_len = len(prefix)
    elif len(prefix) < half_remaining_tokens:
        prefix_len = len(prefix)
    else:
        raise ValueError

    prefix = prefix[-prefix_len:]

    mention_context = prefix + mention + suffix
    mention_start = len(prefix)
    mention_end = mention_start + len(mention)
    mention_context = mention_context[:max_tokens]

    assert mention_start <= max_tokens
    assert mention_end <= max_tokens

    return mention_context, mention_start, mention_end


def get_context(context_words, start_index, end_index, max_tokens):
    " extract mention context with an evenly distributed sliding window"
    start_pos = start_index - max_tokens
    if start_pos < 0:
        start_pos = 0
    prefix = context_words[start_pos: start_index]
    suffix = context_words[end_index + 1: end_index + max_tokens + 1]
    mention = context_words[start_index: end_index + 1]

    assert len(mention) < max_tokens

    remaining_tokens = max_tokens - len(mention)
    half_remaining_tokens = int(math.ceil(1.0 * remaining_tokens / 2))

    # protect the shorter side
    if len(prefix) >= half_remaining_tokens and len(suffix) >= half_remaining_tokens:
        prefix_len = half_remaining_tokens
    elif len(prefix) >= half_remaining_tokens and len(suffix) < half_remaining_tokens:
        prefix_len = remaining_tokens - len(suffix)
        if prefix_len > len(prefix):
            prefix_len = len(prefix)
    elif len(prefix) < half_remaining_tokens:
        prefix_len = len(prefix)
    else:
        raise ValueError

    prefix = prefix[-prefix_len:]

    mention_context = prefix + mention + suffix
    mention_context = " ".join(mention_context[:max_tokens])
    mention_context = normalize_context(mention_context)
    return mention_context

def pad_sequence(tokens, max_len):
    assert len(tokens) <= max_len
    return tokens + [0]*(max_len - len(tokens))

def customized_tokenize(tokenizer, tokens, max_seq_length, mention_start=None,
                        mention_end=None, return_tensor="pt"):
    if type(tokens) ==str:
        tokens = tokenizer.tokenize(tokens, add_special_tokens=True, max_length=max_seq_length, truncation=True)
    else:
        tokens = ["[CLS]"] + tokens + ["[SEP]"]

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # token type ids
    token_type_ids = [0]*len(tokens)
    if mention_start and mention_end:
        for idx in range(mention_start+1, mention_end+1):
            token_type_ids[idx] = 2
    #  set mention span as 2 ["CLS"]
    attention_mask = [1]*len(input_ids)

    input_ids = pad_sequence(input_ids, max_seq_length)
    token_type_ids = pad_sequence(token_type_ids, max_seq_length)
    attention_mask = pad_sequence(attention_mask, max_seq_length)

    if return_tensor == "pt":
        input_ids = torch.LongTensor([input_ids])
        token_type_ids = torch.LongTensor([token_type_ids])
        attention_mask = torch.LongTensor([attention_mask])
    return {"input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask
            }


class EntityLinkingSet(Dataset):
    """Create `TrainingInstance`s from raw text."""
    def __init__(self, document_files, mentions_path, tfidf_candidates_file, max_seq_length,
                 num_candidates, is_training=True,):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.num_candidates = num_candidates
        self.rng = random.Random(12345)
        self.max_seq_length = max_seq_length
        print("max sequence length: {}".format(self.max_seq_length))
        self.is_training = is_training

        self.all_documents = document_files
        self.candidates = load_candidates(tfidf_candidates_file)   # mention_id to candidates
        # mention_id, context_id, label_id, start_idx, end_idx, corpus (domain), label (target documents in candidates)
        self.mentions = self.load_mentions(mentions_path)


    def filter_mention(self, mention):
        mention_id = mention["mention_id"]
        label_document_id = mention["label_document_id"]
        assert mention_id in self.candidates
        cand_document_ids = self.candidates[mention_id]
        # skip this mention if there is no tf-idf candidate
        if not cand_document_ids:
            return None
        # if manually labelled description doc of the mention is not in the noisy tf-idf set, skip this mention
        elif not self.is_training and label_document_id not in cand_document_ids:
            return None
        else:
            return mention

    def load_mentions(self, mention_dir):
        with open(mention_dir, "r", encoding="utf-8") as m:
            mentions = json.load(m)
        print("Loading {} mentions from {}".format(len(mentions), mention_dir))
        return mentions

    def __len__(self):
        return len(self.mentions)

    def __getitem__(self, item):
        return self.create_cl_instances(self.mentions[item])

    def create_cl_instances(self, mention):
        """Creates Next Sentence Prediction Instance for a single document."""

        # Account for [CLS], [SEP]
        max_num_tokens = self.max_seq_length - 2

        # mention and context
        mention_id = mention['mention_id']
        context_document_id = mention['context_document_id']
        label_document_id = mention['label_document_id']
        start_index = mention['start_index']  # start idx in the context doc
        end_index = mention['end_index']  # end idx in the context doc

        context = self.all_documents[context_document_id]['text']

        context_tokens = context.split()
        extracted_mention = context_tokens[start_index: end_index + 1]
        extracted_mention = ' '.join(extracted_mention)
        mention_text = mention['text']
        assert extracted_mention == mention_text

        mention_context, mention_start, mention_end = get_context_tokens(self.tokenizer,
            context_tokens, start_index, end_index, max_num_tokens)

        label_idx = mention['label']        # indices of gts in candidate sets
        input_dicts = customized_tokenize(self.tokenizer, mention_context, self.max_seq_length, mention_start, mention_end)
        label_document = self.all_documents[label_document_id]['text']
        label_document = normalize_context(self.all_documents[label_document_id]['text'])
        label_dicts = customized_tokenize(self.tokenizer, label_document, self.max_seq_length)

        # adding tf-idf candidates (including gt by default)
        cand_document_ids = self.candidates[mention_id]
        if self.is_training:
            # del gt from negative samples
            # cand_document_ids = [cand for cand in cand_document_ids if cand != label_document_id]
            del cand_document_ids[label_idx]

            cand_document_ids = cand_document_ids[:self.num_candidates]     # truncate to num_candidates for cl

        candidates_input_dicts = []
        for cand_document_id in cand_document_ids:
            cand_document = self.all_documents[cand_document_id]['text']
            cand_document = normalize_context(self.all_documents[cand_document_id]['text'])
            cand_dict = customized_tokenize(self.tokenizer, cand_document, self.max_seq_length)

            candidates_input_dicts.append(cand_dict)

        instance = CLInstance(
            mention_dict=input_dicts,
            entity_dict=label_dicts,
            candidate_dicts=candidates_input_dicts,
            mention_id=mention_id,
            label_id=label_document_id,
            label=torch.LongTensor([label_idx]),
            corpus=mention['corpus']
        )
        return instance


class ColBERTDatasetAdapter(EntityLinkingSet):
    """
    Convert dataset format from DPR to ColBERT
    """
    def __init__(self, output_path, document_files, mentions_path, tfidf_candidates_file, max_seq_length,
                 num_candidates, is_training=True, contextualized=True):
        super().__init__(document_files, mentions_path, tfidf_candidates_file, max_seq_length,
                 num_candidates, is_training)
        self.output_path = output_path
        print("Is training: {}".format(is_training))
        self.is_training = is_training
        self.contextualized = contextualized    # whether to use contextualized doc or not
        document_path = document_files[0].split(",")
        self.all_documents = {}
        self.stats_by_domain = {}
        for input_file_path in document_path:
            self.all_documents.update(load_documents(input_file_path))
        print("{} documents loaded".format(len(self.all_documents)))

    def contextualized_document(self, mention):
        # Account for [CLS], [SEP]
        max_num_tokens = self.max_seq_length - 2

        # mention and context
        mention_id = mention['mention_id']
        context_document_id = mention['context_document_id']
        start_index = mention['start_index']  # start idx in the context doc
        end_index = mention['end_index']  # end idx in the context doc

        context = self.all_documents[context_document_id]['text']

        context_tokens = context.split()
        extracted_mention = context_tokens[start_index: end_index + 1]
        extracted_mention = ' '.join(extracted_mention)
        mention_text = mention['text']
        assert extracted_mention == mention_text

        mention_context = get_context(context_tokens, start_index, end_index, max_num_tokens)
        return {"_id": context_document_id, "text": mention_context, "metadata": mention['corpus']}

    def retrieve_context_mention_from_sample(self, sample, context_limit):
        """
        Extract mention and context into a dictionary and generate a query from them with a foundation model
        """
        label_id = sample['label_document_id']
        context_document_id = sample['context_document_id']
        context = self.all_documents[context_document_id]['text']
        start_index = sample['start_index']
        end_index = sample['end_index']
        mention_context = get_context(context.split(" "), start_index, end_index, context_limit + 5)
        entity_context = self.all_documents[label_id]['text']
        mention_context = normalize_context(mention_context)
        entity_context = normalize_context(self.all_documents[label_id]['text'])
        entity_desc = " ".join(entity_context.split(" ")[:context_limit])
        return {"mention_id": label_id, "mention": sample["text"], 
                "context": mention_context, "corpus": sample["corpus"], 
                 "entity": self.all_documents[label_id]['title'], "entity_desc": entity_desc}
    
    def analyse_doc_len(self):
        def _tokenize_thread(buffer):
            for _doc in buffer:
                _d = _doc['metadata']
                if _domain not in stats_by_domain:
                    self.stats_by_domain[_d] = []
                else:
                    self.stats_by_domain[_d].append(len(self.tokenizer.tokenize(_doc['text'])))

        doc_queue = queue.Queue(maxsize=10)
        thread = threading.Thread(target=_tokenize_thread)
        thread.start()

        stats_by_domain = {}
        for d_id, d in self.all_documents.items():
            doc_queue.put(d)
        total_doc_list = []
        for _domain, doc_len_list in self.stats_by_domain.items():
            total_doc_list.extend(doc_len_list)
            stats_by_domain[_domain] = np.mean(doc_len_list)
        print("Average document length: {}".format(np.mean(total_doc_list)))
        print(stats_by_domain)

    def save_corpus(self):
        with open(os.path.join(self.output_path, 'corpus.jsonl'), 'w') as f:
            for d_id, d in self.all_documents.items():
                _id = d['document_id']
                title = d['title']
                text = normalize_context(d['text'])
                metadata = d['metadata']
                assert _id == d_id, "Document id must be the same."
                json.dump({'_id':_id, 'title': title, 'text':text, 'metadata': metadata}, f, separators=(',', ':'))
                f.write('\n')

    def save_queries(self):
        with open(os.path.join(self.output_path, 'test_queries.jsonl'), 'w') as f:
            for m in self.mentions:
                json.dump(self.contextualized_document(m), f, separators=(',', ':'))
                f.write('\n')

    def save_triples(self):
        tsv_name = 'train.tsv' if self.is_training else 'valid.tsv'
        print("Saving {} of sample to {}".format(len(self.mentions), os.path.join(self.output_path, tsv_name)))
        with open(os.path.join(os.path.join(self.output_path, "qrels"), tsv_name), 'w') as f:
            writer = csv.writer(f, delimiter='\t')
            for i, m in enumerate(self.mentions):
                data_dict = self.create_triples_for_current_mention(m)
                neg_samples = data_dict['pid-']
                for neg in neg_samples:
                    writer.writerow([data_dict['qid'], data_dict['pid+'], neg])

    def save_test_pairs(self):
        print("Saving test positive pair to {}/qrels/test.tsv".format(self.output_path))
        tsv_name = 'test.tsv'
        with open(os.path.join("{}/qrels/test.tsv".format(self.output_path), tsv_name), 'w') as f:
            writer = csv.writer(f, delimiter='\t')
            for i, m in enumerate(self.mentions):
                data_dict = self.create_triples_for_current_mention(m)
                writer.writerow([data_dict['qid'], data_dict['pid+'], 1.0])

    def create_triples_for_current_mention(self, mention):
        # mention and context
        mention_id = mention['mention_id']  # mention id only represent id in the query queue
        context_document_id = mention['context_document_id']
        label_document_id = mention['label_document_id']

        # adding tf-idf candidates (including gt by default)
        cand_document_ids = self.candidates[mention_id]
        if label_document_id in cand_document_ids:
            cand_document_ids.remove(label_document_id)
        return {'qid': context_document_id, 'pid+': label_document_id, 'pid-': cand_document_ids[:self.num_candidates]}

    def run_adapter(self, save_corpus=False, save_query=False):
        self.save_triples()
        if save_corpus:
            self.save_corpus()
        if save_query:
            self.save_queries()


def collate(batch_data):
    input_ids = []
    attention_mask = []
    token_type_ids = []

    for sep_dict in batch_data:
        input_ids.append(sep_dict['input_ids'])
        attention_mask.append(sep_dict['attention_mask'])
        token_type_ids.append(sep_dict['token_type_ids'])

    return {"input_ids": torch.cat(input_ids, 0),
            "attention_mask": torch.cat(attention_mask, 0),
            "token_type_ids": torch.cat(token_type_ids, 0),
            }


def compose_collate(batch_cl_data: List[CLInstance]):
    mention_dicts = [cl_data.mention_dict for cl_data in batch_cl_data]
    mention_dicts = collate(mention_dicts)

    label_dicts = [cl_data.entity_dict for cl_data in batch_cl_data]
    label_dicts = collate(label_dicts)

    labels = [cl_data.label for cl_data in batch_cl_data]
    labels = torch.cat(labels, 0)

    mention_ids = [cl_data.mention_id for cl_data in batch_cl_data]
    label_ids = [cl_data.label_id for cl_data in batch_cl_data]

    input_ids = []
    attention_mask = []
    token_type_ids = []

    for cl_data in batch_cl_data:
        cand_dict_list = cl_data.candidate_dicts
        _cand_dict = collate(cand_dict_list)
        input_ids.append(_cand_dict['input_ids'])
        attention_mask.append(_cand_dict['attention_mask'])
        token_type_ids.append(_cand_dict['token_type_ids'])

    return {
        "mention_dicts": mention_dicts,
        "entity_dicts": label_dicts,
        "labels": labels,
        "me_mask": get_label_mask(mention_ids, label_ids),
        "mm_mask": get_mention_mask(mention_ids),
        "candidate_dicts": {"input_ids": torch.stack(input_ids, 0),
                            "attention_mask": torch.stack(attention_mask, 0),
                            "token_type_ids": torch.stack(token_type_ids, 0)
                            }
    }


if __name__ == '__main__':

    parser = ArgumentParser(description='test arguments')
    parser.add_argument("--document-files", nargs="+", default=None,
                        help="Path to train documents json file.")
    parser.add_argument("--train-mentions-file", default='/home/marvinpeng/datasets/data_zip/Fandomwiki/mentions/valid.json',
                        type=str,
                        help="Path to train mentions json file.")
    parser.add_argument("--train-tfidf-candidates-file",
                        default='/home/marvinpeng/datasets/data_zip/Fandomwiki/tfidf_candidates/valid_tfidfs.json', type=str,
                        help="Path to train candidates file retrieved by BM25.")
    parser.add_argument("--test-mentions-file", default='/home/marvinpeng/datasets/data_zip/Fandomwiki/mentions/test.json',
                        type=str,
                        help="Path to train mentions json file.")
    parser.add_argument("--test-tfidf-candidates-file",
                        default='/home/marvinpeng/datasets/data_zip/Fandomwiki/tfidf_candidates/test_tfidfs.json', type=str,
                        help="Path to train candidates file retrieved by BM25.")
    parser.add_argument('--output-path', type=str, default='data/Fandomwiki', help='output path for the adapted data')
    parser.add_argument('--is-training', action='store_true', help='Flag for training')
    parser.add_argument('--save-query', action='store_true', help='Flag for saving query')
    parser.add_argument('--save-corpus', action='store_true', help='Flag for saving corpus')
    parser.add_argument("--max-seq-length", default=64, type=int, help="Maximum sequence length.")
    parser.add_argument("--num-candidates", default=32, type=int, help="Number of tfidf candidates (0-63).")

    args = parser.parse_args()
    print("save query: {}".format(args.save_query))
    print("save corpus: {}".format(args.save_corpus))

    os.makedirs(args.output_path, exist_ok=True)
    train_adapter = ColBERTDatasetAdapter(args.output_path, args.document_files, args.train_mentions_file,
                                          args.train_tfidf_candidates_file, args.max_seq_length, args.num_candidates, is_training=args.is_training)

    train_adapter.run_adapter(save_corpus=args.save_corpus, save_query=args.save_query)

    # test_adapter = ColBERTDatasetAdapter(args.output_path, args.document_files, args.test_mentions_file,
    #                                      args.test_tfidf_candidates_file, args.max_seq_length, args.num_candidates, is_training=False)
    # test_adapter.save_test_pairs()
    # test_adapter.run_adapter(save_corpus=False, save_query=True)