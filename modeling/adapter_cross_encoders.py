import torch
import torch.nn as nn
from collections import OrderedDict
from utils import logger
import os
from transformers import AutoModel, AutoConfig
from transformers.adapters import LoRAConfig


def _deactivate_relevant_gradients(model, trainable_components):
    # turns off the model parameters requires_grad except the trainable bias terms.
    for param in model.parameters():
        param.requires_grad = False
    if trainable_components:
        trainable_components = trainable_components + ['pooler.dense.bias']

    for name, param in model.named_parameters():
        for component in trainable_components:
            if component in name:
                param.requires_grad = True
                break


class CrossEncoder(nn.Module):
    def __init__(self, adapter=False, bitfit=False, lora=False, r=64):
        super().__init__()
        self.config = AutoConfig.from_pretrained('bert-base-uncased')
        self.model = AutoModel.from_pretrained('bert-base-uncased')
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

        if adapter:
            self.model.add_adapter("el_adapter", config='pfeiffer')
            self.model.train_adapter("el_adapter")

        if bitfit:
            _deactivate_relevant_gradients(self.model, ['bias'])

        if lora:
            config = LoRAConfig(r=r, use_gating=False)     # Wx + alpha / r * BAx r = 1, 8, 64
            self.model.add_adapter("el_lora_adapter", config=config)
            self.model.train_adapter("el_lora_adapter")

        total_num_parameters = 0
        for mention_param in self.model.named_parameters():
            total_num_parameters += mention_param[1].numel() if mention_param[1].requires_grad else 0

        print("total number of parameters: {}M".format(total_num_parameters / 1000000))
        all_param = 0
        for _, param in self.named_parameters():
            all_param += param.numel()

        print("Number of training parameters: {}".format(all_param))

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