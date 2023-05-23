import torch.nn as nn
from transformers import BertModel, AutoConfig
from transformers.models.bert import BertModel
import copy
import torch
import torch.nn.functional as F
import os
from collections import OrderedDict
from utils import logger


class EntityLinker(nn.Module):
    """
    Using separate BERT models for query and entity encoding.
    """
    def __init__(self,):
        super(EntityLinker, self).__init__()
        self.config = AutoConfig.from_pretrained('bert-base-uncased')
        self.hidden_size = self.config.hidden_size

        self.mention_encoder = BertModel(config=self.config, add_pooling_layer=False)
        self.mention_encoder.from_pretrained('bert-base-uncased')

        # adding mention span as a new type to token type ids
        old_type_vocab_size = self.config.type_vocab_size
        self.config.type_vocab_size = 3
        new_token_type_embeddings = nn.Embedding(self.config.type_vocab_size, self.config.hidden_size)
        self.mention_encoder._init_weights(new_token_type_embeddings)
        new_token_type_embeddings.weight.data[:old_type_vocab_size, :] = self.mention_encoder.embeddings.token_type_embeddings.weight.data[:old_type_vocab_size, :]
        self.mention_encoder.embeddings.token_type_embeddings = new_token_type_embeddings
        self.entity_encoder = copy.deepcopy(self.mention_encoder)

        all_param = 0
        for _, param in self.named_parameters():
            all_param += param.numel()

        print("Number of training parameters: {}".format(all_param))

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
            return entity_embeddings

        # contrastive learning
        mention_vectors = self.encode(self.mention_encoder, **mention_dicts)
        entity_vectors = self.encode(self.entity_encoder, **entity_dicts)

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
        else:
            candidate_vectors = None
        return {
                "mention_vectors": mention_vectors,
                "candidate_vectors": candidate_vectors,
                "entity_vectors": entity_vectors,
                }

    def compute_logits(self, me_mask, mm_mask, mention_vectors, entity_vectors, candidate_vectors):
        bs = len(mention_vectors)
        cosine = mention_vectors.mm(entity_vectors.t())
        if self.training:
            logits = cosine - torch.zeros_like(cosine, device=cosine.device).fill_diagonal_(self.additive_margin)
        else:
            logits = cosine

        if me_mask.any():           # avoid contrasting over many-to-one mapping
            logits.masked_fill_(me_mask, 1e-4)

        if mm_mask is not None:     # use in-batch query as negative samples
            mm_logits = mention_vectors.mm(mention_vectors.t())
            mm_logits.masked_fill_(mm_mask, 1e-4)
            mm_logits.fill_diagonal_(-100)
            logits = torch.cat([logits, mm_logits], 1)

        negative_logits = None
        if candidate_vectors is not None:
            candidate_vectors = candidate_vectors.view(-1, self.hidden_size)
            bs_seq_len = candidate_vectors.shape[0]
            candidate_vectors = candidate_vectors.unsqueeze(0).expand(bs, bs_seq_len, self.hidden_size)
            negative_logits = torch.matmul(mention_vectors.view(bs, 1, self.hidden_size), candidate_vectors.permute(0, 2, 1)).squeeze(1)

        if negative_logits is not None:
            assert len(logits) == len(negative_logits)
            logits = torch.cat([logits, negative_logits], 1)       # bs, num_cand, hidden_dim
        logits = logits * self.inv_t
        return logits

    @torch.no_grad()
    def predict(self, mention_dicts, candidate_dicts_list, labels):
        mention_vectors = self.encode(self.mention_encoder, **mention_dicts)

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

    def resume_from_checkpoint(self, entity_state_dict, mention_state_dict):
        self.load_pretrained_model(self.mention_encoder, mention_state_dict, strict=False)
        self.load_pretrained_model(self.entity_encoder, entity_state_dict, strict=False)

    @staticmethod
    def load_pretrained_model(encoder, checkpoint_dict, strict):

        new_state_dict = OrderedDict()
        for k, v in checkpoint_dict.items():
            if k.startswith('module.'):
                k = k[len('module.'):]
            new_state_dict[k] = v
        encoder.load_state_dict(new_state_dict, strict=strict)

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


class EntityLinkingPredictor(nn.Module):
    def __init__(self, pretrained_path):
        super().__init__()
        self.model = EntityLinker()
        self.load_pretrained_model(pretrained_path, strict=True)

    def load_pretrained_model(self, checkpoint_path, strict):
        assert os.path.exists(checkpoint_path)
        checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")

        new_state_dict = OrderedDict()
        for k, v in checkpoint_dict.items():
            if k.startswith('module.'):
                k = k[len('module.'):]
            new_state_dict[k] = v
        self.model.load_state_dict(new_state_dict, strict=strict)
        logger.info('Loading model successfully')

    @torch.no_grad()
    def predict(self, mention_dicts, candidate_dicts_list, labels):
        scores, metrics = self.model.predict(mention_dicts, candidate_dicts_list, labels)
        return scores, metrics


