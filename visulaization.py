import json
from transformers import BertTokenizerFast
from modeling.bi_encoders import EntityLinkingPredictor
import torch
import os
import numpy as np


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

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
model = EntityLinkingPredictor('checkpoints/bi_encoders/model_best.ckpt').cuda()
encoder = model.model.entity_encoder
with open("data/In-Zeshel/domain2samples.jsonl", "r") as f:
    domain2samples = json.load(f)

batch_size = 1024
for domain, sentences in domain2samples.items():
    total_embeds = []
    for idx in range(0, len(sentences), batch_size):
        batch_sentences = sentences[idx*batch_size:(idx+1)*batch_size]
        if not batch_sentences:
            break
        input_dict = tokenizer.batch_encode_plus(batch_sentences, max_length=64, truncation=True,
                                                 padding='max_length', return_tensors='pt')

        with torch.no_grad():
            input_dict = move_to_cuda(input_dict)
            batch_embeds = encoder(**input_dict).last_hidden_state.mean(dim=1)
            total_embeds.append(batch_embeds)

    total_embeds = torch.cat(total_embeds, 0).cpu().numpy()
    np.save("./visualization/{}.npy".format(domain), total_embeds, )
