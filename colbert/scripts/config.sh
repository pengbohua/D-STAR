#!/bin/bash

export dataset=$1
export psl=64
export step=-1
export SIMILARITY=cosine
export checkpoint=colbert-150000

export DATA_DIR=../beir_eval/datasets/$dataset/

export CHECKPOINT=../checkpoints/msmarco.psg.l2/checkpoints/colbert-150000.dnn

# Path where preprocessed collection and queries are present
# Collections and queries are preprocessed from jsonl to tsv files
export COLLECTION="../data/${dataset}/collections.tsv"
export QUERIES="../data/${dataset}/queries.tsv"

# Path to store the faiss index and run output
export INDEX_ROOT="output/index"
export OUTPUT_DIR="output/output"
export EXPERIMENT=$checkpoint.$step.$dataset
# Path to store the rankings file
#export RANKING_DIR="output/rankings/${checkpoint}/${step}/${dataset}"
export RANKING_DIR="output/rankings/colbert-150000/query-generation"
# Num of partitions in for IVPQ Faiss index (You must decide yourself)
export NUM_PARTITIONS=96

# Some Index Name to store the faiss index
export INDEX_NAME=index.$checkpoint.$step.$dataset

if [ "$dataset" = "msmarco" ]; then
    SPLIT="dev"
else
    SPLIT="test"
fi

echo $SPLIT
