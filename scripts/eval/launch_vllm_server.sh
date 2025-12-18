#!/bin/bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_VISIBLE_DEVICES=${1:-0,1,2,3,4,5,6,7}
tensor_parallel_size=${2:-8}
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, tensor_parallel_size=$tensor_parallel_size"

python -m vllm.entrypoints.openai.api_server \
	--model models/Qwen2.5-VL-7B-Instruct \
	--trust-remote-code \
	--served-model-name Qwen2.5-VL-7B-Instruct \
	--limit-mm-per-prompt image=16,video=0 \
	--gpu-memory-utilization 0.8 \
	--tensor-parallel-size $tensor_parallel_size