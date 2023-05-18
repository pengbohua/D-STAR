##########################################################################################
# 1. Feature Extraction: Extract claim (token) embeddings using ColBERT checkpoint #
##########################################################################################
python -m colbert.query_generation.extract_zeshel_embs --amp --mask-punctuation \
--doc_maxlen 300 \
--bsize 256 \
--checkpoint ../checkpoints/msmarco_colbert/colbert-150000.dnn \
--root output \
--queries ../data/scifact/mention_entity.json \
--experiment colbert-feature-extraction \
--dataset-type scifact \
--similarity cosine

##########################################################################################
# 2. Similarity Estimation: Compute cosine similarity K-NNs for each claim  #
#    Build the graph: build a graph from K-NNs and a path considering region size #
##########################################################################################

python -m colbert.query_generation.dstar \
--feature-path colbert/query_generation/scifact_colbert/scifacts.npy \
--output-knn colbert/query_generation/scifact_colbert/query_knn10.npy \
--output-knn-distance colbert/query_generation/scifact_colbert/query_distance_knn10.npy \
--claim-to-mention ../data/scifact/fid2mention_id.json \
--output-query-graph ../data/scifact/query_graph.jsonl \
--output-query-path ../data/scifact/query_path.jsonl \
--distance-threshold 0.65 \
--region-size 5

##########################################################################################
# 3. Demonstrative query generation: Generate a sequence of queries from the path  #
##########################################################################################
python -m colbert.query_generation.dstar_queries
