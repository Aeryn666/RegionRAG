#!/bin/bash

# retrieval args
dataset_name=${1:-infovqa}
dataset_path=${2:-data_dir/VisRAG/VisRAG-Ret-Test-InfoVQA}
bbox_score_method=max
batch_size=16

retrieval_model_path=models/RegionRet
retrieval_model_name=$(echo $retrieval_model_path | rev | cut -d'/' -f1 | rev)
eval_save_root=work_dirs/eval_output/${retrieval_model_name}/${dataset_name}

CUDA_VISIBLE_DEVICES=0 python scripts/eval/retrieval.py \
    --model_path $retrieval_model_path \
    --dataset_name $dataset_name \
    --dataset_path $dataset_path \
    --topks 1,2,5,10,100 \
    --bbox_score_method $bbox_score_method \
    --eval_save_root $eval_save_root \
    --batch_size $batch_size \
    --force_inference True


# generation args
generate_model_path=models/Qwen2.5-VL-7B-Instruct
generate_model_name=$(echo $generate_model_path | rev | cut -d'/' -f1 | rev)


# # using transformers to deploy
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port 29666 \
#     --nproc_per_node 4 \
#     scripts/eval/generate.py \
#     --model_path $generate_model_path \
#     --deploy_type transformers \
#     --max_new_tokens 40 \
#     --dataset_name $dataset_name \
#     --dataset_path $dataset_path \
#     --eval_save_root $eval_save_root \
#     --force_generate false \
#     --batch_size 1 \
#     --vllm_tensor_parallel_size 1 \
#     --api_num_processes 30


# # using vllm api deploy, run bash launvh_vllm_server.sh first
# python scripts/eval/generate.py \
#     --model_path $generate_model_path \
#     --deploy_type api \
#     --max_new_tokens 40 \
#     --dataset_name $dataset_name \
#     --dataset_path $dataset_path \
#     --eval_save_root $eval_save_root \
#     --force_generate false \
#     --batch_size 1 \
#     --vllm_tensor_parallel_size 4 \
#     --topk_image 4 \
#     --api_num_processes 10