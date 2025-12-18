YAML_PATH=$1
YAML_NAME=$(echo $YAML_PATH | rev | cut -d'/' -f1 | rev)
YAML_NAME="${YAML_NAME%.*}"

mkdir -p work_dirs/${YAML_NAME}

torchrun \
    --nnodes 1 \
    --nproc_per_node 8 \
    --master_addr=127.0.0.1 \
    --master_port=23456 \
    --node_rank 0 \
    scripts/train/train_colbert.py ${YAML_PATH} \
    2>&1 | tee -a work_dirs/${YAML_NAME}/training.log

