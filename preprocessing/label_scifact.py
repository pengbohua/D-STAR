from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification
import json
from tqdm import tqdm

tokenizer = AutoTokenizer.from_pretrained("d4data/biomedical-ner-all")
model = AutoModelForTokenClassification.from_pretrained("d4data/biomedical-ner-all")

pipe = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple") # pass device=0 if using gpu
# print(pipe("0-dimensional biomaterials lack inductive properties."))
ner_results = []
with open("data/scifact/queries.jsonl", "r") as f:
    lines = f.readlines()
    queries = []
    for l in lines:
        queries.append(json.loads(l))

# with open("/home/marvinpeng/wikidata5m/peng/EntityLinkingForFandom/data/scifact/queries.jsonl", "r") as f:
#     queries = json.load(f)

qid2fid = {}
for i, q in enumerate(queries):
    qid2fid[i] = q

# with open("/home/marvinpeng/wikidata5m/peng/EntityLinkingForFandom/scifact/val_meta_qid2fid.json", "w") as f:
#     json.dump(qid2fid, f, indent=4)
for q in tqdm(queries):
    results = pipe(q['text'])
    new_results = []
    for res in results:
        res['score'] = float(res['score'])
        new_results.append(res)
    ner_results.append(new_results)

with open("/home/marvinpeng/wikidata5m/peng/EntityLinkingForFandom/scifact/train_mention.jsonl", "w") as f:
    json.dump(ner_results, f)

