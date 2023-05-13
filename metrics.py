import torch

@torch.no_grad()
def compute_metric(logits: torch.tensor, labels: torch.tensor):
    batch_scores = logits.reshape(-1, 64)
    cs, num_cand = batch_scores.shape

    batch_labels = torch.stack(labels.chunk(cs), 0)
    batch_labels = batch_labels.argmax(1, keepdim=True)  # cs, 1
    batch_labels = batch_labels.expand(cs, num_cand)

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
    metrics = {k: round(v / cs, 4) for k, v in metrics.items()}
    metrics["chunk_size"] = cs
    return metrics


@torch.no_grad()
def accuracy(logits, labels):
    predictions = torch.ge(logits, 0).long().squeeze(1)
    return torch.sum(torch.eq(predictions, labels)) / len(labels)