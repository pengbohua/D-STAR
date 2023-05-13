import torch.nn as nn
from transformers import BertModel, AutoConfig
from transformers.models.bert import BertModel
import copy
import torch
import torch.nn.functional as F
import os
from collections import OrderedDict


class EntityLinker(nn.Module):
    """
    Using shared BERT encoder for contrastive entity linking.
    """
    def __init__(self, pretrained_model_path):
        super(EntityLinker, self).__init__()
        self.config = AutoConfig.from_pretrained(pretrained_model_path)
        self.hidden_size = self.config.hidden_size
        self.entity_encoder = BertModel(config=self.config, add_pooling_layer=False)
        self.load_pretrained_model(pretrained_model_path)

        # adding mention span as a new type to token type ids
        old_type_vocab_size = self.config.type_vocab_size
        self.config.type_vocab_size = 3
        new_token_type_embeddings = nn.Embedding(self.config.type_vocab_size, self.config.hidden_size)
        self.entity_encoder._init_weights(new_token_type_embeddings)
        new_token_type_embeddings.weight.data[:old_type_vocab_size, :] = self.entity_encoder.embeddings.token_type_embeddings.weight.data[:old_type_vocab_size, :]
        self.entity_encoder.embeddings.token_type_embeddings = new_token_type_embeddings

        self.pooling = 'mean'
        self.additive_margin = 0.0
        self.inv_t = torch.tensor(20.0, requires_grad=False)

    def encode(self, encoder, input_ids, attention_mask, token_type_ids):
        outputs = encoder(input_ids=input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids,
                          return_dict=True
                          )
        last_hidden_state = outputs.last_hidden_state
        embeddings = self.pool_output(last_hidden_state, attention_mask)
        return embeddings

    def pool_output(self, last_hidden_state, attention_mask):
        if self.pooling == 'cls':
            output_vector = last_hidden_state[:, 0, :]
        elif self.pooling == 'max':
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).long()
            last_hidden_state[input_mask_expanded == 0] = -100
            output_vector = torch.max(last_hidden_state, 1)[0]
        elif self.pooling == 'mean':
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-4)
            output_vector = sum_embeddings / sum_mask
        else:
            print('Unknown pooling mode: {}'.format(self.pooling))
            raise ValueError

        output_vector = F.normalize(output_vector, dim=1)
        return output_vector

    def forward(self, entity_dicts, mention_dicts=None,  candidate_dict_list=None):
        if mention_dicts is None:
            assert (not self.training and candidate_dict_list is None)
            with torch.no_grad():
                entity_embeddings = self.encode(self.entity_encoder, **entity_dicts)
                # entity_embeddings = self.encode(self.entity_encoder, **entity_dicts)
            return entity_embeddings

        bs = len(mention_dicts['input_ids'])

        # contrastive learning
        mention_vectors = self.encode(self.entity_encoder, **mention_dicts)
        entity_vectors = self.encode(self.entity_encoder, **entity_dicts)
        neg_mention_vectors = self.encode(self.entity_encoder, **mention_dicts)

        # entity_vectors = self.encode(self.entity_encoder, **entity_dicts)
        # neg_mention_vectors = self.encode(self.entity_encoder, **mention_dicts)

        candidate_vectors = []
        if candidate_dict_list is not None:
            for i in range(len(mention_vectors)):
                cur_candidate_dict = {"input_ids": candidate_dict_list["input_ids"][i],
                                "attention_mask": candidate_dict_list["attention_mask"][i],
                                "token_type_ids": candidate_dict_list["token_type_ids"][i],
                }
                cand_vec = self.encode(self.entity_encoder, **cur_candidate_dict)  # N negative sample for a single mention
                candidate_vectors.append(cand_vec)

        if len(candidate_vectors) != 0:
            candidate_vectors = torch.stack(candidate_vectors, 0)       # bs, num_cand, hidden_dim
            negative_logits = torch.matmul(mention_vectors.view(bs, 1, self.hidden_size), candidate_vectors.permute(0, 2, 1)).squeeze(1)
        else:
            negative_logits = None

        return {
                "mention_vectors": mention_vectors,
                "negative_mention_vectors": neg_mention_vectors,
                "entity_vectors": entity_vectors,
                "negative_logits": negative_logits,
                }

    def compute_logits(self, me_mask, mm_mask, mention_vectors, negative_mention_vectors, entity_vectors, negative_logits):
        cosine = mention_vectors.mm(entity_vectors.t())
        if self.training:
            logits = cosine - torch.zeros_like(cosine, device=cosine.device).fill_diagonal_(self.additive_margin)
        else:
            logits = cosine

        logits.masked_fill_(me_mask, -1e4)

        if mm_mask is not None:
            mm_logits = mention_vectors.mm(negative_mention_vectors.t())
            mm_logits.masked_fill_(mm_mask, -1e4)
            # mm_logits.fill_diagonal_(-1e4)
            logits = torch.cat([logits, mm_logits], 1)

        if negative_logits is not None:
            assert len(logits) == len(negative_logits)
            logits = torch.cat([logits, negative_logits], 1)       # bs, num_cand, hidden_dim

        logits = logits * self.inv_t
        return logits

    @torch.no_grad()
    def predict(self, mention_dicts, candidate_dicts_list, labels):
        mention_vectors = self.encode(self.entity_encoder, **mention_dicts)

        candidate_vectors = []

        assert len(candidate_dicts_list["input_ids"]) == len(mention_vectors)

        for i in range(len(mention_vectors)):
            cur_candidate_dict = {"input_ids": candidate_dicts_list["input_ids"][i],
                                  "attention_mask": candidate_dicts_list["attention_mask"][i],
                                  "token_type_ids": candidate_dicts_list["token_type_ids"][i],
                                  }

            cand_vec = self.encode(self.entity_encoder, **cur_candidate_dict)  # N negative sample for a single mention
            candidate_vectors.append(cand_vec)

        candidate_vectors = torch.stack(candidate_vectors, 0).permute(0, 2, 1)

        bs = len(mention_vectors)
        mention_vectors = mention_vectors.view(bs, 1, self.hidden_size)
        scores = torch.matmul(mention_vectors, candidate_vectors).squeeze(1)
        metrics = self.compute_metric(scores, labels)
        return scores, metrics

    def load_pretrained_model(self, checkpoint_path):
        assert os.path.exists(checkpoint_path)
        checkpoint_dict = torch.load(checkpoint_path+"/pytorch_model.bin", map_location="cpu")

        new_state_dict = OrderedDict()
        for k, v in checkpoint_dict.items():
            if k.startswith('module.'):
                k = k[len('module.'):]
            new_state_dict[k] = v
        self.entity_encoder.load_state_dict(new_state_dict, strict=False)

    @staticmethod
    def compute_metric(batch_scores: torch.tensor, labels: torch.tensor):
        bs, num_cand = batch_scores.shape
        batch_labels = labels.unsqueeze(1)

        batch_sorted_score, batch_sorted_indices = torch.sort(batch_scores, dim=-1, descending=True)
        target_rank = torch.nonzero(batch_sorted_indices.eq(batch_labels).long(), as_tuple=False)

        assert target_rank.size(0) == batch_sorted_score.size(0)

        mean_rank = 0
        mrr = 0
        hit1, hit3, hit10 = 0, 0, 0
        for idx in range(batch_scores.size(0)):
            idx_rank = target_rank[idx].tolist()
            assert idx_rank[0] == idx
            cur_rank = idx_rank[1]

            # 0-based -> 1-based
            cur_rank += 1
            mean_rank += cur_rank
            mrr += 1.0 / cur_rank
            hit1 += 1 if cur_rank <= 1 else 0
            hit3 += 1 if cur_rank <= 3 else 0
            hit10 += 1 if cur_rank <= 10 else 0

        metrics = {'mean_rank': mean_rank, 'mrr': mrr, 'hit1': hit1, 'hit3': hit3, 'hit10': hit10}
        metrics = {k: round(v / bs, 4) for k, v in metrics.items()}
        return metrics




