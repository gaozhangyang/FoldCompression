# BioNemo Framework

BioNemo Framework is a modular and extensible platform for bioinformatics and computational biology workflows. It aims to simplify the development, deployment, and management of complex data analysis pipelines.


## Usage

```bash
export PYTHONPATH="/nfs_beijing/kubeflow-user/zhangyang_2024/workspace/StructCompression/"

python ./task/main.py \
    --cluster-path ./data/train_keys_subset.msgpack \
    --database-path ./data/afdb_rep_mem_debug.db \
    --precision="bf16-mixed" \
    --num-gpus 1 \
    --num-nodes 1 \
    --num-steps 50000 \
    --val-check-interval 50 \
    --result-dir ./results/esm2-demo/ \
    --max-seq-length 1024 \
    --resume-if-exists \
    --limit-val-batches 4 \
    --micro-batch-size 8 \
    --num-layers 12 \
    --hidden-size 480 \
    --num-attention-head 20 \
    --ffn-hidden-size 1920 \
    --tensor-model-parallel-size 1 \
    --create-tensorboard-logger
```
