import torch
from modeling.bi_encoders import EntityLinker


class PrefixEncoder(torch.nn.Module):
    '''
    The torch.nn model to encode the prefix
    Input shape: (batch-size, prefix-length)
    Output shape: (batch-size, prefix-length, 2*layers*hidden)
    '''
    def __init__(self, config):
        super().__init__()
        self.prefix_projection = config.prefix_projection
        if self.prefix_projection:
            # Use a two-layer MLP to encode the prefix
            self.embedding = torch.nn.Embedding(config.pre_seq_len*2, config.hidden_size)
            self.trans = torch.nn.Sequential(
                torch.nn.Linear(config.hidden_size, config.prefix_hidden_size),
                torch.nn.Tanh(),
                torch.nn.Linear(config.prefix_hidden_size, config.num_hidden_layers * 2 * config.hidden_size)
            )
        else:
            self.embedding = torch.nn.Embedding(config.pre_seq_len, config.num_hidden_layers * 2 * config.hidden_size)

    def forward(self, prefix: torch.Tensor):
        if self.prefix_projection:
            prefix_tokens = self.embedding(prefix)
            past_key_values = self.trans(prefix_tokens)
        else:
            past_key_values = self.embedding(prefix)
        return past_key_values


class PrefixEntityLinker(EntityLinker):
    """
    Using separate BERT models for query & entity encoding and prefix tuning for parameter efficient tuning.
    """
    def __init__(self, prefix_seq_len=180, prefix_projection=True):
        super().__init__()

        self.pre_seq_len = prefix_seq_len
        print("prefix length", self.pre_seq_len)
        self.prefix_tokens = torch.arange(self.pre_seq_len*2).long()

        self.config.pre_seq_len = self.pre_seq_len
        self.config.prefix_hidden_size = self.config.hidden_size
        self.config.prefix_projection = prefix_projection
        self.prefix_encoder = PrefixEncoder(self.config)
        self.dropout = torch.nn.Dropout(self.config.hidden_dropout_prob)
        print(self.prefix_encoder)


        bert_param_num = 0
        for query_param, doc_param in zip(self.mention_encoder.parameters(), self.entity_encoder.parameters()):
            query_param.requires_grad = False
            doc_param.requires_grad = False
            bert_param_num += query_param.numel()
            bert_param_num += doc_param.numel()

        total_num_parameters = 0
        for mention_param in self.prefix_encoder.named_parameters():
            total_num_parameters += mention_param[1].numel() if mention_param[1].requires_grad else 0

        print("total number of parameters: {}M".format(total_num_parameters / 1000000))
        all_param = 0
        for _, param in self.named_parameters():
            all_param += param.numel()

        print("Number of training parameters: {}".format(all_param))

    def get_prompt(self, batch_size, prefix_tokens):
        """Get prompt tokens to meet input requirements.

        Args:
            batch_size (_type_): _description_

        Returns:
            _type_: _description_
        """
        prefix_tokens = prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.entity_encoder.device)
        past_key_values = self.prefix_encoder(prefix_tokens)

        past_key_values = past_key_values.view(
            batch_size,
            self.pre_seq_len,
            self.config.num_hidden_layers * 2,
            self.config.num_attention_heads,
            self.config.hidden_size // self.config.num_attention_heads
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2) # layers, bs, num_heads, seq_len, hid_dim
        return past_key_values, prefix_tokens

    def _encode(self, encoder, input_ids, attention_mask, token_type_ids, past_key_values):
        prefix_mask = torch.ones(len(attention_mask), self.pre_seq_len).long().to(attention_mask.device)
        prefix_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        # attend to past key values
        outputs = encoder(input_ids=input_ids,
                          attention_mask=prefix_mask,
                          token_type_ids=token_type_ids,
                          past_key_values=past_key_values,)
        last_hidden_state = outputs.last_hidden_state
        embeddings = self.pool_output(last_hidden_state, attention_mask)
        return embeddings

    @torch.no_grad()
    def predict_ent_embedding(self, encoder, tail_token_ids, tail_mask, tail_token_type_ids, past_key_values, **kwargs) -> dict:
        ent_vectors = self._encode(encoder,
                                   input_ids=tail_token_ids,
                                   attention_mask=tail_mask,
                                   token_type_ids=tail_token_type_ids,
                                   past_key_values=past_key_values,
                                   )
        return ent_vectors.detach()

    def forward(self, entity_dicts, mention_dicts=None,  candidate_dict_list=None):
        batchsize = len(entity_dicts['input_ids'])
        entity_past_key_values, _ = self.get_prompt(batchsize, self.prefix_tokens[:self.pre_seq_len])
        if mention_dicts is None:
            assert (not self.training and candidate_dict_list is None)
            with torch.no_grad():
                entity_embeddings = self.predict_ent_embedding(self.entity_encoder, past_key_values=entity_past_key_values, **entity_dicts)
            return entity_embeddings

        mention_past_key_values, _ = self.get_prompt(batchsize, self.prefix_tokens[self.pre_seq_len:])
        # contrastive learning
        mention_vectors = self._encode(self.mention_encoder, past_key_values=mention_past_key_values, **mention_dicts)
        entity_vectors = self._encode(self.entity_encoder, past_key_values=entity_past_key_values, **entity_dicts)

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

