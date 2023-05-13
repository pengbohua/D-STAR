from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import tensor
import collections
import json
import math
import random
from transformers import AutoConfig, AutoTokenizer
import json
from dataclasses import dataclass, field
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple
from utils import logger, load_documents, load_candidates
import os
from preprocessing.utils import normalize_context


@dataclass
class NSPInstance(object):
    """A single set of features for EL as a next sentence prediction task"""
    input_dicts: List[Dict]
    label_ids : tensor
    mention_id : str
    doc_ids : List[str]
    corpus : str


def get_context_tokens(tokenizer, context_tokens, start_index, end_index, max_tokens):
    " extract mention context with an evenly distributed sliding window"
    start_pos = start_index - max_tokens
    if start_pos < 0:
        start_pos = 0
    prefix = ' '.join(context_tokens[start_pos: start_index])
    suffix = ' '.join(context_tokens[end_index + 1: end_index + max_tokens + 1])
    prefix = tokenizer.tokenize(prefix)
    suffix = tokenizer.tokenize(suffix)
    # mention = tokenizer.tokenize(' '.join(context_tokens[start_index: end_index + 1]))
    mention = tokenizer.tokenize(' '.join(context_tokens[start_index: end_index + 1]))[:max_tokens-1]   # post hoc removal

    assert len(mention) < max_tokens, "mention starting from {} ending from {}: {}".format(start_index, end_index + 1, context_tokens[start_index: end_index + 1])

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

def pad_sequence(tokens, max_len):
    assert len(tokens) <= max_len
    return tokens + [0]*(max_len - len(tokens))

def customized_tokenize(tokenizer, token_a, text_pair_b, text_pair_b_max_len, max_seq_length, mention_start=None,
                        mention_end=None, return_tensor="pt"):
    text_pair_b_max_len = min(text_pair_b_max_len, max_seq_length - len(token_a) - 2)
    token_pair_b = tokenizer.tokenize(text=text_pair_b)[:text_pair_b_max_len]
    tokens = ["[CLS]"] + token_a + ["[SEP]"] + token_pair_b

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    token_type_ids = [0]*(len(token_a)+2)+[1]*len(token_pair_b)
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
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.num_candidates = num_candidates
        self.rng = random.Random(12345)
        self.max_seq_length = max_seq_length
        print("max sequence length: {}".format(self.max_seq_length))
        self.is_training = is_training

        self.all_documents = document_files
        self.candidates = load_candidates(tfidf_candidates_file)   # mention_id to candidates

        self.mentions = self.load_mentions(mentions_path)       # mention_id, context_id, label_id, start_idx, end_idx

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
        logger.info("Loading {} mentions from {}".format(len(mentions), mention_dir))
        return mentions

    def reserve_topk_tf_idf_candidates(self):
        if not self.is_training:
            topk_candidates_dict = {}
            for cand_key, cand_document_ids in self.candidates.items():
                topk_candidates_dict[cand_key] = cand_document_ids[:self.num_candidates]

            self.candidates = topk_candidates_dict

    def __len__(self):
        return len(self.mentions)

    def __getitem__(self, item):
        return self.create_nsp_instances(self.mentions[item])

    def create_nsp_instances(self, mention):
        """Creates Next Sentence Prediction Instance for a single document."""

        # Account for [CLS], [SEP]
        max_num_tokens = self.max_seq_length - 2

        mention_length = int(max_num_tokens / 2)
        cand_length = max_num_tokens - mention_length

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
            context_tokens, start_index, end_index, mention_length)
        mention_context = normalize_context(" ".join(mention_context))
        mention_context = mention_context.split()[:max_num_tokens]
        label_idx = mention['label']
        # label_id is now in candidates (all)
        cand_document_ids = self.candidates[mention_id]

        # adding tf-idf candidates as negative samples with label_id
        if self.is_training:
            cand_document_ids = cand_document_ids[:self.num_candidates]
            label_ids = torch.zeros(self.num_candidates).long()

            if label_idx > self.num_candidates-1:
                cand_document_ids.insert(0, label_document_id)
                cand_document_ids = cand_document_ids[:self.num_candidates]
                label_ids[0].fill_(1)
                label_ids = label_ids.long()
            else:
                label_ids[label_idx].fill_(1)

        else:
            # label for nsp prediction
            label_ids = torch.zeros(len(cand_document_ids)).long()
            label_ids[label_idx].fill_(1)

        doc_input_dicts = []
        doc_ids = []
        for cand_document_id in cand_document_ids:
            cand_document = self.all_documents[cand_document_id]['text']
            cand_document = normalize_context(cand_document)
            doc_dict = customized_tokenize(self.tokenizer, mention_context, cand_document, cand_length,
                                               self.max_seq_length)
            doc_input_dicts.append(doc_dict)
            doc_ids.append(cand_document_id)

        instance = NSPInstance(
            input_dicts=doc_input_dicts,
            label_ids=label_ids,
            mention_id=mention_id,
            doc_ids=doc_ids,
            corpus=mention['corpus']
        )
        return instance


def collate(batch_data: List[NSPInstance]) -> dict:
    input_ids = []
    attention_mask = []
    token_type_ids = []
    labels = []

    for el_instance in batch_data:
        input_dicts = el_instance.input_dicts
        for doc_dict in input_dicts:
            input_ids.append(doc_dict['input_ids'])
            attention_mask.append(doc_dict['attention_mask'])
            token_type_ids.append(doc_dict['token_type_ids'])
        labels.append(el_instance.label_ids)

    return {
        "input_ids": torch.cat(input_ids, 0),
        "attention_mask": torch.cat(attention_mask, 0),
        "token_type_ids": torch.cat(token_type_ids, 0),
        'labels': torch.cat(labels, 0)
    }