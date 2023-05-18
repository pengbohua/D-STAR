python -m supervision.self_training \
--rankings colbert/output/query_generation/ranking.tsv \
--output colbert/output/selftraining.epoch.1 \
--positives 1 \
--depth+ 3 \
--depth- 65 \
--cutoff- 64