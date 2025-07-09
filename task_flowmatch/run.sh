export PYTHONPATH="/nfs_beijing/kubeflow-user/zhangyang_2024/workspace/StructCompression/"
export https_proxy=http://172.30.1.70:18000;export http_proxy=http://172.30.1.70:18000
cd /nfs_beijing/kubeflow-user/zhangyang_2024/workspace/StructCompression




CUDA_VISIBLE_DEVICES=6  python ./task_flowmatch/main.py \
    --cluster-path /nfs_beijing_os/linlinchao/afdb/rep_mem_v2/afdb_rep_mem-cluster.msgpack \
    --database-path /nfs_beijing_os/linlinchao/afdb/rep_mem_v2/afdb_rep_mem.db \
    --data-splits '9990, 5, 5' \
    --precision="bf16" \
    --num-gpus 1 \
    --num-nodes 1 \
    --num-steps 1000000 \
    --val-check-interval 1000 \
    --result-dir ./results/struct_compress_FM/ \
    --min-seq-length 512 \
    --max-seq-length 512 \
    --resume-if-exists \
    --limit-val-batches 4 \
    --micro-batch-size 8 \
    --num-layers 12 \
    --hidden-size 480 \
    --num-attention-head 20 \
    --ffn-hidden-size 1920 \
    --tensor-model-parallel-size 1 \
    --create-tensorboard-logger \
    --wandb-offline 0 \
    --lr 1e-4 \
    --prefix_len 32 \
    --experiment-name baseline_prefix32_len512_FM_1M \
    --log-every-n-steps 100 


CUDA_VISIBLE_DEVICES=7  python ./task_flowmatch/main.py \
    --cluster-path /nfs_beijing_os/linlinchao/afdb/rep_mem_v2/afdb_rep_mem-cluster.msgpack \
    --database-path /nfs_beijing_os/linlinchao/afdb/rep_mem_v2/afdb_rep_mem.db \
    --data-splits '9990, 5, 5' \
    --precision="bf16" \
    --num-gpus 1 \
    --num-nodes 1 \
    --num-steps 1000000 \
    --val-check-interval 1000 \
    --result-dir ./results/struct_compress_FM/ \
    --min-seq-length 512 \
    --max-seq-length 512 \
    --resume-if-exists \
    --limit-val-batches 4 \
    --micro-batch-size 8 \
    --num-layers 12 \
    --hidden-size 480 \
    --num-attention-head 20 \
    --ffn-hidden-size 1920 \
    --tensor-model-parallel-size 1 \
    --create-tensorboard-logger \
    --wandb-offline 0 \
    --lr 1e-5 \
    --prefix_len 32 \
    --experiment-name baseline_prefix32_len512_FM_1M_lr1e-5 \
    --log-every-n-steps 100 

CUDA_VISIBLE_DEVICES=3  python ./task_flowmatch/main.py \
    --cluster-path /nfs_beijing_os/linlinchao/afdb/rep_mem_v2/afdb_rep_mem-cluster.msgpack \
    --database-path /nfs_beijing_os/linlinchao/afdb/rep_mem_v2/afdb_rep_mem.db \
    --data-splits '9990, 5, 5' \
    --precision="bf16" \
    --num-gpus 1 \
    --num-nodes 1 \
    --num-steps 1000000 \
    --val-check-interval 1000 \
    --result-dir ./results/struct_compress_FM/ \
    --min-seq-length 512 \
    --max-seq-length 512 \
    --resume-if-exists \
    --limit-val-batches 4 \
    --micro-batch-size 8 \
    --num-layers 12 \
    --hidden-size 480 \
    --num-attention-head 20 \
    --ffn-hidden-size 1920 \
    --tensor-model-parallel-size 1 \
    --create-tensorboard-logger \
    --wandb-offline 0 \
    --lr 1e-4 \
    --prefix_len 32 \
    --experiment-name baseline_prefix32_len512_FM_1M_predX0 \
    --log-every-n-steps 100 

CUDA_VISIBLE_DEVICES=2  python ./task_flowmatch/main.py \
    --cluster-path /nfs_beijing_os/linlinchao/afdb/rep_mem_v2/afdb_rep_mem-cluster.msgpack \
    --database-path /nfs_beijing_os/linlinchao/afdb/rep_mem_v2/afdb_rep_mem.db \
    --data-splits '9990, 5, 5' \
    --precision="bf16" \
    --num-gpus 1 \
    --num-nodes 1 \
    --num-steps 1000000 \
    --val-check-interval 1000 \
    --result-dir ./results/struct_compress_FM/ \
    --min-seq-length 512 \
    --max-seq-length 512 \
    --resume-if-exists \
    --limit-val-batches 4 \
    --micro-batch-size 8 \
    --num-layers 12 \
    --hidden-size 480 \
    --num-attention-head 20 \
    --ffn-hidden-size 1920 \
    --tensor-model-parallel-size 1 \
    --create-tensorboard-logger \
    --wandb-offline 0 \
    --lr 1e-5 \
    --prefix_len 32 \
    --experiment-name baseline_prefix32_len512_FM_1M_predX0_lr1e-5 \
    --log-every-n-steps 100 


CUDA_VISIBLE_DEVICES=7  python ./task_flowmatch/main.py \
    --cluster-path /nfs_beijing_os/linlinchao/afdb/rep_mem_v2/afdb_rep_mem-cluster.msgpack \
    --database-path /nfs_beijing_os/linlinchao/afdb/rep_mem_v2/afdb_rep_mem.db \
    --data-splits '9990, 5, 5' \
    --precision="bf16" \
    --num-gpus 1 \
    --num-nodes 1 \
    --num-steps 1000000 \
    --val-check-interval 1000 \
    --result-dir ./results/struct_compress_FM/ \
    --min-seq-length 512 \
    --max-seq-length 512 \
    --resume-if-exists \
    --limit-val-batches 4 \
    --micro-batch-size 8 \
    --num-layers 12 \
    --hidden-size 480 \
    --num-attention-head 20 \
    --ffn-hidden-size 1920 \
    --tensor-model-parallel-size 1 \
    --create-tensorboard-logger \
    --wandb-offline 0 \
    --lr 1e-4 \
    --prefix_len 32 \
    --experiment-name baseline_prefix32_len512_FM_1M_linearXdec \
    --log-every-n-steps 100 