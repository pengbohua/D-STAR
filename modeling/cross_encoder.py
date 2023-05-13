import torch
import torch.nn as nn
from transformers import AutoConfig, BertModel 
from collections import OrderedDict
from utils import logger
import os


class CrossEncoder(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.config = AutoConfig.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        # adding mention span as a new type to token type ids
        old_type_vocab_size = self.config.type_vocab_size
        self.config.type_vocab_size = 3
        new_token_type_embeddings = nn.Embedding(self.config.type_vocab_size, self.config.hidden_size)
        self.model._init_weights(new_token_type_embeddings)
        new_token_type_embeddings.weight.data[:old_type_vocab_size, :] = self.model.embeddings.token_type_embeddings.weight.data[:old_type_vocab_size, :]
        self.model.embeddings.token_type_embeddings = new_token_type_embeddings
        self.model.predict_head = nn.Linear(self.config.hidden_size, 1)
        self.model = self.model.cuda()
        self.t = 1

    def forward(self, **batch_dict):
        outputs = self.model(**batch_dict)
        h = outputs.last_hidden_state[:, 0, :]
        logits = self.model.predict_head(h) * self.t
        return logits

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

    def resume_from_checkpoint(self, model_state_dict):
        self.load_pretrained_model(model_state_dict, strict=True)


class EntityLinkingPredictor(CrossEncoder):
    def __init__(self, pretrained_model_path):
        super().__init__( )
        self.resume_from_checkpoint(pretrained_model_path)
        
    @torch.no_grad()
    def forward(self, **batch_dict):
        outputs = self.model(**batch_dict)
        h = outputs.last_hidden_state[:, 0, :]
        logits = self.model.predict_head(h)
        return logits