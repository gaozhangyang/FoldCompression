export PYTHONPATH="/nfs_beijing/kubeflow-user/zhangyang_2024/workspace/StructCompression/"
export https_proxy=http://172.30.1.70:18000;export http_proxy=http://172.30.1.70:18000
cd /nfs_beijing/kubeflow-user/zhangyang_2024/workspace/StructCompression

CUDA_VISIBLE_DEVICES=1  python ./task/main.py \
    --cluster-path /nfs_beijing_os/linlinchao/afdb/rep_mem_v2/afdb_rep_mem-cluster.msgpack \
    --database-path /nfs_beijing_os/linlinchao/afdb/rep_mem_v2/afdb_rep_mem.db \
    --data-splits '9990, 5, 5' \
    --precision="bf16" \
    --num-gpus 1 \
    --num-nodes 1 \
    --num-steps 100000 \
    --val-check-interval 1000 \
    --result-dir ./results/struct_compress/ \
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
    --prefix_len 16 \
    --experiment-name baseline_prefix16_len512 \
    --log-every-n-steps 100

CUDA_VISIBLE_DEVICES=2  python ./task/main.py \
    --cluster-path /nfs_beijing_os/linlinchao/afdb/rep_mem_v2/afdb_rep_mem-cluster.msgpack \
    --database-path /nfs_beijing_os/linlinchao/afdb/rep_mem_v2/afdb_rep_mem.db \
    --data-splits '9990, 5, 5' \
    --precision="bf16" \
    --num-gpus 1 \
    --num-nodes 1 \
    --num-steps 100000 \
    --val-check-interval 1000 \
    --result-dir ./results/struct_compress/ \
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
    --experiment-name baseline_prefix32_len512 \
    --log-every-n-steps 100


CUDA_VISIBLE_DEVICES=3  python ./task/main.py \
    --cluster-path /nfs_beijing_os/linlinchao/afdb/rep_mem_v2/afdb_rep_mem-cluster.msgpack \
    --database-path /nfs_beijing_os/linlinchao/afdb/rep_mem_v2/afdb_rep_mem.db \
    --data-splits '9990, 5, 5' \
    --precision="bf16" \
    --num-gpus 1 \
    --num-nodes 1 \
    --num-steps 100000 \
    --val-check-interval 1000 \
    --result-dir ./results/struct_compress/ \
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
    --prefix_len 8 \
    --experiment-name baseline_prefix8_len512 \
    --log-every-n-steps 100


CUDA_VISIBLE_DEVICES=4  python ./task/main.py \
    --cluster-path /nfs_beijing_os/linlinchao/afdb/rep_mem_v2/afdb_rep_mem-cluster.msgpack \
    --database-path /nfs_beijing_os/linlinchao/afdb/rep_mem_v2/afdb_rep_mem.db \
    --data-splits '9990, 5, 5' \
    --precision="bf16" \
    --num-gpus 1 \
    --num-nodes 1 \
    --num-steps 100000 \
    --val-check-interval 1000 \
    --result-dir ./results/struct_compress/ \
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
    --experiment-name baseline_prefix32_len512_dec1 \
    --log-every-n-steps 100 \
    --dec_layers 1

CUDA_VISIBLE_DEVICES=5  python ./task/main.py \
    --cluster-path /nfs_beijing_os/linlinchao/afdb/rep_mem_v2/afdb_rep_mem-cluster.msgpack \
    --database-path /nfs_beijing_os/linlinchao/afdb/rep_mem_v2/afdb_rep_mem.db \
    --data-splits '9990, 5, 5' \
    --precision="bf16" \
    --num-gpus 1 \
    --num-nodes 1 \
    --num-steps 100000 \
    --val-check-interval 1000 \
    --result-dir ./results/struct_compress/ \
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
    --experiment-name baseline_prefix32_len512_enc15_dec1 \
    --log-every-n-steps 100 \
    --dec_layers 1 \
    --enc_layers 15


CUDA_VISIBLE_DEVICES=4  python ./task/main.py \
    --cluster-path /nfs_beijing_os/linlinchao/afdb/rep_mem_v2/afdb_rep_mem-cluster.msgpack \
    --database-path /nfs_beijing_os/linlinchao/afdb/rep_mem_v2/afdb_rep_mem.db \
    --data-splits '9990, 5, 5' \
    --precision="bf16" \
    --num-gpus 1 \
    --num-nodes 1 \
    --num-steps 1000000 \
    --val-check-interval 1000 \
    --result-dir ./results/struct_compress/ \
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
    --experiment-name baseline_prefix32_len512_dec1_continue1M \
    --log-every-n-steps 100 \
    --dec_layers 1 \
    --custom_checkpoint_path /nfs_beijing/kubeflow-user/zhangyang_2024/workspace/StructCompression/results/struct_compress/baseline_prefix32_len512_dec1/checkpoints/epoch=0-step=94999-consumed_samples=760000.0/model.pt

CUDA_VISIBLE_DEVICES=5  python ./task/main.py \
    --cluster-path /nfs_beijing_os/linlinchao/afdb/rep_mem_v2/afdb_rep_mem-cluster.msgpack \
    --database-path /nfs_beijing_os/linlinchao/afdb/rep_mem_v2/afdb_rep_mem.db \
    --data-splits '9990, 5, 5' \
    --precision="bf16" \
    --num-gpus 1 \
    --num-nodes 1 \
    --num-steps 1000000 \
    --val-check-interval 1000 \
    --result-dir ./results/struct_compress/ \
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
    --experiment-name baseline_prefix32_len512_dec1_1M \
    --log-every-n-steps 100 \
    --dec_layers 1 


CUDA_VISIBLE_DEVICES=0  python ./task/main.py \
    --cluster-path /nfs_beijing_os/linlinchao/afdb/rep_mem_v2/afdb_rep_mem-cluster.msgpack \
    --database-path /nfs_beijing_os/linlinchao/afdb/rep_mem_v2/afdb_rep_mem.db \
    --data-splits '9990, 5, 5' \
    --precision="bf16" \
    --num-gpus 1 \
    --num-nodes 1 \
    --num-steps 1000000 \
    --val-check-interval 1000 \
    --result-dir ./results/struct_compress/ \
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
    --prefix_len 16 \
    --experiment-name baseline_prefix16_len512_dec1_1M \
    --log-every-n-steps 100 \
    --dec_layers 1 


CUDA_VISIBLE_DEVICES=1  python ./task/main.py \
    --cluster-path /nfs_beijing_os/linlinchao/afdb/rep_mem_v2/afdb_rep_mem-cluster.msgpack \
    --database-path /nfs_beijing_os/linlinchao/afdb/rep_mem_v2/afdb_rep_mem.db \
    --data-splits '9990, 5, 5' \
    --precision="bf16" \
    --num-gpus 1 \
    --num-nodes 1 \
    --num-steps 1000000 \
    --val-check-interval 1000 \
    --result-dir ./results/struct_compress/ \
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
    --prefix_len 8 \
    --experiment-name baseline_prefix8_len512_dec1_1M \
    --log-every-n-steps 100 \
    --dec_layers 1 

CUDA_VISIBLE_DEVICES=3  python ./task/main.py \
    --cluster-path /nfs_beijing_os/linlinchao/afdb/rep_mem_v2/afdb_rep_mem-cluster.msgpack \
    --database-path /nfs_beijing_os/linlinchao/afdb/rep_mem_v2/afdb_rep_mem.db \
    --data-splits '9990, 5, 5' \
    --precision="bf16" \
    --num-gpus 1 \
    --num-nodes 1 \
    --num-steps 1000000 \
    --val-check-interval 1000 \
    --result-dir ./results/struct_compress/ \
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
    --prefix_len 4 \
    --experiment-name baseline_prefix4_len512_dec1_1M \
    --log-every-n-steps 100 \
    --dec_layers 1 
    # --cluster-path /nfs_beijing_os/linlinchao/afdb/rep_mem_v2/afdb_rep_mem-cluster.msgpack \
    # --database-path /nfs_beijing_os/linlinchao/afdb/rep_mem_v2/afdb_rep_mem.db \

    # --cluster-path ./data/train_keys_subset.msgpack \
    # --database-path ./data/afdb_rep_mem_debug.db \


CUDA_VISIBLE_DEVICES=5  python ./task/main.py \
    --cluster-path /nfs_beijing_os/linlinchao/afdb/rep_mem_v2/afdb_rep_mem-cluster.msgpack \
    --database-path /nfs_beijing_os/linlinchao/afdb/rep_mem_v2/afdb_rep_mem.db \
    --data-splits '9990, 5, 5' \
    --precision="bf16" \
    --num-gpus 1 \
    --num-nodes 1 \
    --num-steps 1000000 \
    --val-check-interval 1000 \
    --result-dir ./results/struct_compress/ \
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
    --experiment-name baseline_prefix32_len512_dec1_1M_bs32 \
    --log-every-n-steps 100 \
    --dec_layers 1 

CUDA_VISIBLE_DEVICES=7  python ./task/main.py \
    --cluster-path /nfs_beijing_os/linlinchao/afdb/rep_mem_v2/afdb_rep_mem-cluster.msgpack \
    --database-path /nfs_beijing_os/linlinchao/afdb/rep_mem_v2/afdb_rep_mem.db \
    --data-splits '9990, 5, 5' \
    --precision="bf16" \
    --num-gpus 1 \
    --num-nodes 1 \
    --num-steps 1000000 \
    --val-check-interval 1000 \
    --result-dir ./results/struct_compress/ \
    --min-seq-length 512 \
    --max-seq-length 512 \
    --resume-if-exists \
    --limit-val-batches 4 \
    --micro-batch-size 64 \
    --num-layers 12 \
    --hidden-size 480 \
    --num-attention-head 20 \
    --ffn-hidden-size 1920 \
    --tensor-model-parallel-size 1 \
    --create-tensorboard-logger \
    --wandb-offline 0 \
    --lr 1e-4 \
    --prefix_len 32 \
    --experiment-name baseline_prefix32_len512_dec1_1M_bs64 \
    --log-every-n-steps 100 \
    --dec_layers 1 

CUDA_VISIBLE_DEVICES=1  python ./task/main.py \
    --cluster-path /nfs_beijing_os/linlinchao/afdb/rep_mem_v2/afdb_rep_mem-cluster.msgpack \
    --database-path /nfs_beijing_os/linlinchao/afdb/rep_mem_v2/afdb_rep_mem.db \
    --data-splits '9990, 5, 5' \
    --precision="bf16" \
    --num-gpus 1 \
    --num-nodes 1 \
    --num-steps 1000000 \
    --val-check-interval 1000 \
    --result-dir ./results/struct_compress/ \
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
    --experiment-name baseline_prefix32_len512_dec1_1M_bs32_run3 \
    --log-every-n-steps 100 \
    --dec_layers 1 


# CUDA_VISIBLE_DEVICES=1,2,3,4,5,7 
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
    --micro-batch-size 16 \
    --num-layers 12 \
    --hidden-size 480 \
    --num-attention-head 20 \
    --ffn-hidden-size 1920 \
    --tensor-model-parallel-size 1 \
    --create-tensorboard-logger \
    --wandb-offline 0 \
    --lr 1e-5 \
    --prefix_len 32 \
    --experiment-name DecLayer12_new_megatron_partial_mask \
    --log-every-n-steps 100 \
    --dec_layers 12 \
    --infer_feats 0 \
    --custom_checkpoint_path /nfs_beijing/kubeflow-user/zhangyang_2024/workspace/StructCompression/results/struct_compress/baseline_prefix32_len512_dec1_1M_bs32_run3/checkpoints/epoch=0-step=999999-consumed_samples=64192032.0-last.pt


