import faiss
import numpy as np
import torch
import argparse
import random
import json

random.seed(1234)
FAISS_AVAILABLE = False
visited = set()


def region_grow(graph, seed, min_dist, region_size=5):
    global visited
    region_list = []
    region = [seed]
    to_check = [seed]
    while to_check:
        node = to_check.pop(0)
        if node in visited:
            continue
        visited.add(node)
        for neighbor in graph[node]:
            neighbor_id, dist = neighbor[0], neighbor[1]
            if neighbor_id not in visited and dist > min_dist:      # cosine similarity
                region.append(neighbor_id)
                to_check.append(neighbor_id)

        if len(region) >= region_size:
            region_list.append(region)
            region = []

    if region:
        region_list.append(region)
    unvisited = set(graph.keys()) - visited
    return region_list, unvisited


def get_nn_dist(emb, query, knn):
    """
    Compute the average distance of the `knn` nearest neighbors
    for a given set of embeddings and queries.
    Use Faiss if available.
    """

    if FAISS_AVAILABLE:

        if hasattr(faiss, 'StandardGpuResources'):
            # gpu mode
            res = faiss.StandardGpuResources()      # set gpu resource
            config = faiss.GpuIndexFlatConfig()     # config for gpu
            config.device = 0
            # print("GPU resource: {}".format(res))
            index = faiss.GpuIndexFlatIP(res, emb.shape[1], config)     # index for brutal force search (no invert indexing)
        else:
            # cpu mode
            index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb)      # input numpy array of doc embeddings to indexer
        distances, all_k_neighbours = index.search(query, knn)     # dist: N, k; labels: N, k
        return distances, all_k_neighbours
    else:
        bs = 1024
        emb = torch.from_numpy(emb).cuda()
        query = torch.from_numpy(query).cuda()

        all_distances = []
        all_k_neighbours = []
        emb = emb.transpose(0, 1).contiguous()
        for i in range(0, query.shape[0], bs):
            distances = query[i:i + bs].mm(emb)
            best_distances, k_neighbours = distances.topk(knn, dim=1, largest=True, sorted=True)
            all_distances.append(best_distances.cpu())
            all_k_neighbours.append(k_neighbours.cpu())
        all_distances = torch.cat(all_distances)
        all_k_neighbours = torch.cat(all_k_neighbours)
        return all_distances.numpy(), all_k_neighbours.numpy()


def sample_path_from_graph(query_graph, dist_threshold, region_size, init_seed='185'):
    """
    Sample a path with a region size controlled region growing algorithm.
    @param query_graph:
    @param dist_threshold: distance threshold (minimum similarity) to include the next query into the current session
    @param region_size: region size to control the length of a query session
    @param init_seed: initial seed to start sampling
    @return:
    """
    # path length does not grow with region size, success.
    total_path_list, fresh_nodes = region_grow(query_graph, init_seed, dist_threshold, region_size)    # start from centroids
    while fresh_nodes:
        new_seed = random.sample(fresh_nodes, 1)[0]
        new_region_list, fresh_nodes = region_grow(query_graph, new_seed, dist_threshold)
        total_path_list.extend(new_region_list)
    return total_path_list


def post_process_path(total_path):
    # clean up revisited node (leaf nodes) from the path list
    filtered_visited = set()
    filtered_paths = []
    for path in total_path:
        pruned_path = []
        for node in path:
            if node not in filtered_visited:
                filtered_visited.add(node)
                pruned_path.append(node)
        if pruned_path:
            filtered_paths.append(pruned_path)
    num_nodes = 0
    for r_list in filtered_paths:
        num_nodes += len(r_list)
    return filtered_paths, num_nodes


def get_labels_from_scores(knn_scores):
    return None


def get_csls(batch_q_d_pairs, all_docs, all_query, k=5):
    """
    CSLS for pseudo labelling cleaning
    Args:
        all_docs:
        all_query:
        k:

    Returns:

    """

    dist1, _ = get_nn_dist(all_docs, all_query, k)
    average_dist1 = dist1.mean(0)
    dist2, _ = get_nn_dist(all_query, all_docs, k)
    average_dist2 = dist2.mean(0)
    average_dist1 = torch.from_numpy(average_dist1).type_as(all_docs)
    average_dist2 = torch.from_numpy(average_dist2).type_as(all_query)
    # queries / scores
    query = all_query[batch_q_d_pairs[:, 0]]
    scores = query.mm(all_docs.transpose(0, 1))
    scores.mul_(2)
    scores.sub_(average_dist1[batch_q_d_pairs[:, 0]][:, None])  # sub dist_avg1 in a row wise fashion
    scores.sub_(average_dist2[None, :])  # sub dist_avg2 in a column wise fashion

    get_labels_from_scores(scores)


def convert_faiss_id_to_qid(all_k_neighbours, distances, fid2qid):
    assert len(all_k_neighbours) == len(distances)
    query_graph = {}
    for i in range(len(all_k_neighbours)):
        qid = fid2qid[str(i)]
        k_neighbours = all_k_neighbours[i]
        dists = distances[i]
        k_neighbours_qids = [(fid2qid[str(neighbor)], float(_dist)) for neighbor, _dist in zip(k_neighbours, dists)]
        query_graph[qid] = k_neighbours_qids
    return query_graph


def build_graph_from_knns(args):
    scifact_tensors = np.load(args.feature_path)
    distances, all_k_neighbours = get_nn_dist(scifact_tensors, scifact_tensors, knn=11)
    all_k_neighbours = all_k_neighbours[:, 1:]
    distances = distances[:, 1:]
    np.save(args.output_knn, all_k_neighbours[:, 1:])
    np.save(args.output_knn_distance, distances[:, 1:])

    with open(args.claim_to_mention, "r") as f:
        fid_to_mentiond_ids = json.load(f)

    query_graph = convert_faiss_id_to_qid(all_k_neighbours, distances, fid_to_mentiond_ids)
    return query_graph


parser = argparse.ArgumentParser(description='Similarity estimation to output k-nn for each claims')
parser.add_argument('--feature-path', required=True, type=str, default="colbert/query_generation/scifact_colbert/scifacts.npy",
                    help='Path to passages set .tsv file')
parser.add_argument('--output-knn', required=True, type=str, default="colbert/query_generation/scifact_colbert/query_knn10.npy",
                    help='output path of the knn neighbors of the current claims')
parser.add_argument('--output-knn-distance', required=True, type=str, default="colbert/query_generation/scifact_colbert/query_distance_knn10.npy",
                    help='output distance of the knn neighbors of the current claims')
parser.add_argument('--claim-to-mention', required=True, type=str, default="../datasets/scifact/fid2mention_id.json",
                    help='Claim ids to mentions')
parser.add_argument('--output-query-graph', required=True, type=str, default="../datasets/scifact/query_graph.jsonl",
                    help='query graph built from knns')
parser.add_argument('--output-query-path', required=True, type=str, default="../datasets/scifact/query_path.jsonl",
                    help='query path sampled from the graph')
parser.add_argument('--distance-threshold', required=True, type=float, default=0.65,
                    help='Claim ids to mentions')
parser.add_argument('--region-size', required=True, type=int, default=5,
                    help='Claim ids to mentions')
args = parser.parse_args()

if __name__ == "__main__":
    # build a graph from deep embeddings
    question_graph = build_graph_from_knns(args)
    # with open(args.output_query_graph, 'w') as f:
    #     json.dump(question_graph, f, indent=4)

    # sample a path from the graph
    filtered_path = sample_path_from_graph(question_graph, args.distance_threshold, args.region_size)
    # with open("args.output_query_path", "w") as f:
    #     json.dump(args.output_query_path, f)

    with open(args.output_query_path, "r") as f:  # load a sampled path for demo (188 / 1109)
        query_path = json.load(f)
    filterer_path, num_nodes = post_process_path(filtered_path)
    print("Saving sampled path to {}".format(args.output_query_path))
    with open(args.output_query_path, "w") as f:
        json.dump(filtered_path, f)