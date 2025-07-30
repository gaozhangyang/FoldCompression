
#镜像: harbor.biomap-int.com/foldtoken/docker:bionemo

export PYTHONPATH="/nfs_beijing/kubeflow-user/zhangyang_2024/workspace/StructCompression/"
export https_proxy=http://172.30.1.70:18000;export http_proxy=http://172.30.1.70:18000
cd /nfs_beijing/kubeflow-user/zhangyang_2024/workspace/StructCompression


python ./task/main.py \
    --cluster-path /nfs_beijing_os/linlinchao/afdb/rep_mem_v2/afdb_rep_mem-cluster.msgpack \
    --database-path /nfs_beijing_os/linlinchao/afdb/rep_mem_v2/afdb_rep_mem.db \
    --data-splits '9990, 5, 5' \
    --precision="bf16" \
    --num-gpus 8 \
    --num-nodes 1 \
    --num-steps 1000000 \
    --val-check-interval 1000 \
    --result-dir ./results/struct_compress/ \
    --min-seq-length 512 \
    --max-seq-length 512 \
    --resume-if-exists \
    --limit-val-batches 10 \
    --micro-batch-size 32 \
    --num-layers 12 \
    --hidden-size 480 \
    --num-attention-head 20 \
    --ffn-hidden-size 1920 \
    --tensor-model-parallel-size 1 \
    --create-tensorboard-logger \
    --wandb-offline 0 \
    --lr 1e-5 \
    --prefix_len 32 \
    --experiment-name DecLayer12_svckpt4 \
    --log-every-n-steps 100 \
    --dec_layers 12 \
    --infer_feats 0 \
    --custom_checkpoint_path /nfs_beijing/kubeflow-user/zhangyang_2024/workspace/StructCompression/results/struct_compress/baseline_prefix32_len512_dec1_1M_bs32_run3/checkpoints/epoch=0-step=999999-consumed_samples=64192032.0-last.pt
