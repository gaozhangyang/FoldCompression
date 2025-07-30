export PYTHONPATH="/nfs_beijing/kubeflow-user/zhangyang_2024/workspace/StructCompression/"
export https_proxy=http://172.30.1.70:18000;export http_proxy=http://172.30.1.70:18000
cd /nfs_beijing/kubeflow-user/zhangyang_2024/workspace/StructCompression




python ./task_inversefolding/main.py \
    --cluster-path /nfs_beijing_os/linlinchao/afdb/rep_mem_v2/afdb_rep_mem-cluster.msgpack \
    --database-path /nfs_beijing_os/linlinchao/afdb/rep_mem_v2/afdb_rep_mem.db \
    --data-splits '9990, 5, 5' \
    --precision="bf16" \
    --num-gpus 8 \
    --num-nodes 1 \
    --num-steps 500000 \
    --val-check-interval 1000 \
    --result-dir ./results/task_inversefolding/ \
    --min-seq-length 1024 \
    --max-seq-length 1024 \
    --resume-if-exists \
    --limit-val-batches 10 \
    --micro-batch-size 16 \
    --num-layers 12 \
    --hidden-size 480 \
    --num-attention-head 20 \
    --ffn-hidden-size 1920 \
    --tensor-model-parallel-size 1 \
    --create-tensorboard-logger \
    --wandb-offline 0 \
    --lr 1e-4 \
    --prefix_len 0 \
    --experiment-name EncLayer12 \
    --log-every-n-steps 100 \
    --enc_layers 12 \
    --infer_feats 0 
    
    # \
    # --infer_feats 0 \
    # --custom_checkpoint_path /nfs_beijing/kubeflow-user/zhangyang_2024/workspace/StructCompression/results/struct_compress/baseline_prefix32_len512_dec1_1M_bs32_run3/checkpoints/epoch=0-step=999999-consumed_samples=64192032.0-last.pt


