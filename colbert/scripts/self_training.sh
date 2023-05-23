python -m supervision.self_training \
--rankings output/scifact_query_generation/ranking.tsv \
--output output/self-training \
--positives 1 \
--depth+ 3 \
--depth- 65 \
--cutoff- 64 \
--epochs 1 \
--overwrite