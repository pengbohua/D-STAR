from typing import Dict
import tqdm


def compute_hit_scores(qrels: Dict, result: Dict):
    print("Evaluating query relations")
    mrr = 0
    hit1, hit3, hit10 = 0, 0, 0
    total_queries = len(qrels)
    for qid, doc_dict in tqdm.tqdm(qrels.items()):
        try:
            curr_top100 = result[qid]
        except KeyError:
            continue
        curr_top100 = [k for k, v in sorted(curr_top100.items(), key=lambda x: x[1], reverse=True)]
        label_doc_id = tuple(doc_dict.keys())[0]
        if label_doc_id in curr_top100:
            curr_rank = curr_top100.index(label_doc_id) + 1  # ranks are 1 based
            inv_rank = 1.0 / curr_rank
        else:
            curr_rank = len(curr_top100) + 1
            inv_rank = 1.0 / curr_rank

        mrr += inv_rank
        hit1 += 1 if curr_rank <= 1 else 0
        hit3 += 1 if curr_rank <= 3 else 0
        hit10 += 1 if curr_rank <= 10 else 0

    mrr = mrr / total_queries
    hit1 = hit1 / total_queries
    hit3 = hit3 / total_queries
    hit10 = hit10 / total_queries
    metrics = {'mrr': mrr, '@hit1': hit1, '@hit3': hit3, '@hit10': hit10}
    return metrics

