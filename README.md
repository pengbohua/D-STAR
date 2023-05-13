# D-STAR: Demonstrative Self-Training for Source-free Domain Adaptation of Entity Linking with Foundation Models
## Overview
This repository contains the code of D-STAR (ACM MM 2023 submission 2615) and [Fandomwiki](https://mega.nz/folder/8KEmnbxK#0QKy0QEK-u9Z84hFogf8dw) dataset to evaluate source-free domain adaptation. In this work, we present D-STAR, a framework for solving unsupervised entity linking problems using Demonstrative Self-Training and source-free domain adaptation. 
## Methods
In Figure![image](./images/teaser.pdf)Our approach utilizes few-shot examples to prompt a foundation model to generate factoid context-related questions for mention-entity pairs. The order of these examples is determined by a sampled path from a graph encoded by the retriever. We then directly adapt the retrieval model to the generated query and labels retrieved entity documents with its previous knowledge, aided by a pseudo label denoising strategy. Our group contrastive learning strategy shares negative samples within subgraphs. The updated model recomputes distances within the unvisited graph and optimizes the demonstration priority queue for the next self-training cycle. Our demonstrative self-training strategy updates question generation and question answering simultaneously **without accessing source domain data**. 

## Quick Start

1. Install requirements
```bash
pip install -r requirements.txt
```
2. Clone the master branch:
```bash
git clone -b master https://github.com/pengbohua/EntityLinkingForFandom/tree/master --depth 1
cd D-STAR
```

## Data
unzip the datasets
```bash

unzip data.zip
```

## Evaluate on Fandomwiki
```bash
bash scripts/eval_fandomwiki.sh
```
## Evaluate on Zeshel
```bash
bash scripts/eval_zeshel.sh
```
