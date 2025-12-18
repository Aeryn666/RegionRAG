#!/bin/bash
export CUDA_VISIBLE_DEVICES=7

torchrun \
  --nnodes 1 \
  --nproc_per_node 1 \
  --master_addr 127.0.0.1 \
  --master_port 16666 \
  --node_rank 0 \
  scripts/train/train_colbert.py \
  scripts/configs/pali/train_colpali_model.yaml