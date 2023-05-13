import json
import torch
import re
import string

def normalize_context(s):
    """
    remove stopwords, punctuations and whitespaces in docs
    """
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

class EntityLinkingDict:
    def __init__(self):
        self.mention2entity = json.load(open("data/mention2entity.json", "r", encoding="utf-8"))
        self.entity2mention = json.load(open("data/entity2mention.json", "r", encoding="utf-8"))
        self.num_mention = len(self.mention2entity.keys())
        self.num_entity = len(self.entity2mention.keys())

    def get_ancestor(self, mention: str):
        return self.mention2entity[mention]

    def get_neighbours(self, mention: str):
        neighbours = self.entity2mention[self.get_ancestor(mention)[0]]
        neighbours.remove(mention)
        return neighbours

entity_dict = EntityLinkingDict()

def get_label_mask(row_mention, col_entity):
    assert len(row_mention) == len(col_entity)
    me_mask = torch.zeros(len(row_mention), len(row_mention))

    for i in range(len(row_mention)):
        row_id = col_entity[i]
        for j in range(i+1, len(col_entity)):
            if col_entity[j] == row_id:
                me_mask[i][j].fill_(1)
                me_mask[j][i].fill_(1)
    return me_mask.bool()

def get_mention_mask(row_mention):
    mm_mask = torch.zeros(len(row_mention), len(row_mention))

    for i in range(len(row_mention)):
        row_id = row_mention[i]
        neighbours = entity_dict.get_neighbours(row_id)
        for j in range(i+1, len(row_mention)):
            if row_mention[j] in neighbours:
                mm_mask[i][j].fill_(1)
                mm_mask[j][i].fill_(1)
    return mm_mask.bool()


