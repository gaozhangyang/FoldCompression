export PYTHONPATH="/nfs_beijing/kubeflow-user/zhangyang_2024/workspace/StructCompression/"
export https_proxy=http://172.30.1.70:18000;export http_proxy=http://172.30.1.70:18000
cd /nfs_beijing/kubeflow-user/zhangyang_2024/workspace/StructCompression



python ./task_flowmatch/main.py \
    --cluster-path /nfs_beijing_os/linlinchao/afdb/rep_mem_v2/afdb_rep_mem-cluster.msgpack \
    --database-path /nfs_beijing_os/linlinchao/afdb/rep_mem_v2/compression_data.lmdb \
    --data-splits '9990, 5, 5' \
    --precision="bf16" \
    --num-gpus 1 \
    --num-nodes 1 \
    --num-steps 1000000 \
    --val-check-interval 10000 \
    --result-dir ./results/struct_compress_FM_debug/ \
    --min-seq-length 512 \
    --max-seq-length 512 \
    --resume-if-exists \
    --limit-val-batches 4 \
    --micro-batch-size 32 \
    --num-layers 12 \
    --hidden-size 480 \
    --num-attention-head 20 \
    --ffn-hidden-size 1920 \
    --tensor-model-parallel-size 1 \
    --create-tensorboard-logger \
    --wandb-offline 0 \
    --lr 1e-4 \
    --prefix_len 32 \
    --experiment-name latentFM_30M \
    --log-every-n-steps 100 


