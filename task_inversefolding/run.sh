export PYTHONPATH="/mnt/shared-storage-user/gaozhangyang/workspace/FoldCompression"




python ./task_inversefolding/main.py \
    --cluster-path /mnt/shared-storage-user/beam/gaozhangyang/dataset/afdb_rep_mem-cluster.msgpack \
    --database-path /mnt/shared-storage-user/beam/gaozhangyang/dataset/afdb_rep_mem.db \
    --data-splits '9990, 5, 5' \
    --precision="bf16" \
    --num-gpus 4 \
    --num-nodes 1 \
    --num-steps 100000 \
    --val-check-interval 5000 \
    --result-dir ./results/task_inversefolding/ \
    --min-seq-length 1024 \
    --max-seq-length 1024 \
    --resume-if-exists \
    --limit-val-batches 10 \
    --micro-batch-size 32 \
    --num-layers 12 \
    --hidden-size 480 \
    --num-attention-head 20 \
    --ffn-hidden-size 1920 \
    --tensor-model-parallel-size 1 \
    --create-tensorboard-logger \
    --wandb-offline 1 \
    --lr 1e-3 \
    --prefix_len 0 \
    --experiment-name ailab_IF_enc12_100k \
    --log-every-n-steps 100 \
    --enc_layers 12 \
    --infer_feats 0 
    

CUDA_VISIBLE_DEVICES=0 python ./task_inversefolding/main.py \
    --cluster-path /mnt/shared-storage-user/beam/gaozhangyang/dataset/afdb_rep_mem-cluster.msgpack \
    --database-path /mnt/shared-storage-user/beam/gaozhangyang/dataset/afdb_rep_mem.db \
    --data-splits '9990, 5, 5' \
    --precision="bf16" \
    --num-gpus 1 \
    --num-nodes 1 \
    --num-steps 100000 \
    --val-check-interval 5000 \
    --result-dir ./results/task_inversefolding/ \
    --min-seq-length 512 \
    --max-seq-length 512 \
    --resume-if-exists \
    --limit-val-batches 10 \
    --micro-batch-size 8 \
    --num-layers 12 \
    --hidden-size 480 \
    --num-attention-head 20 \
    --ffn-hidden-size 1920 \
    --tensor-model-parallel-size 1 \
    --create-tensorboard-logger \
    --wandb-offline 1 \
    --lr 1e-3 \
    --prefix_len 0 \
    --experiment-name ailab_IF_enc12_100k_nn8 \
    --log-every-n-steps 100 \
    --enc_layers 12 \
    --dec_layers 12 \
    --infer_feats 0 \
    --nn_neighbors 8


CUDA_VISIBLE_DEVICES=1 python ./task_inversefolding/main.py \
    --cluster-path /mnt/shared-storage-user/beam/gaozhangyang/dataset/afdb_rep_mem-cluster.msgpack \
    --database-path /mnt/shared-storage-user/beam/gaozhangyang/dataset/afdb_rep_mem.db \
    --data-splits '9990, 5, 5' \
    --precision="bf16" \
    --num-gpus 1 \
    --num-nodes 1 \
    --num-steps 100000 \
    --val-check-interval 5000 \
    --result-dir ./results/task_inversefolding/ \
    --min-seq-length 512 \
    --max-seq-length 512 \
    --resume-if-exists \
    --limit-val-batches 10 \
    --micro-batch-size 8 \
    --num-layers 12 \
    --hidden-size 480 \
    --num-attention-head 20 \
    --ffn-hidden-size 1920 \
    --tensor-model-parallel-size 1 \
    --create-tensorboard-logger \
    --wandb-offline 1 \
    --lr 1e-3 \
    --prefix_len 0 \
    --experiment-name ailab_IF_enc12_100k_nn16 \
    --log-every-n-steps 100 \
    --enc_layers 12 \
    --dec_layers 12 \
    --infer_feats 0 \
    --nn_neighbors 16


CUDA_VISIBLE_DEVICES=2 python ./task_inversefolding/main.py \
    --cluster-path /mnt/shared-storage-user/beam/gaozhangyang/dataset/afdb_rep_mem-cluster.msgpack \
    --database-path /mnt/shared-storage-user/beam/gaozhangyang/dataset/afdb_rep_mem.db \
    --data-splits '9990, 5, 5' \
    --precision="bf16" \
    --num-gpus 1 \
    --num-nodes 1 \
    --num-steps 100000 \
    --val-check-interval 5000 \
    --result-dir ./results/task_inversefolding/ \
    --min-seq-length 512 \
    --max-seq-length 512 \
    --resume-if-exists \
    --limit-val-batches 10 \
    --micro-batch-size 8 \
    --num-layers 12 \
    --hidden-size 480 \
    --num-attention-head 20 \
    --ffn-hidden-size 1920 \
    --tensor-model-parallel-size 1 \
    --create-tensorboard-logger \
    --wandb-offline 1 \
    --lr 1e-3 \
    --prefix_len 0 \
    --experiment-name ailab_IF_enc12_100k_nn32 \
    --log-every-n-steps 100 \
    --enc_layers 12 \
    --dec_layers 12 \
    --infer_feats 0 \
    --nn_neighbors 32

CUDA_VISIBLE_DEVICES=3 python ./task_inversefolding/main.py \
    --cluster-path /mnt/shared-storage-user/beam/gaozhangyang/dataset/afdb_rep_mem-cluster.msgpack \
    --database-path /mnt/shared-storage-user/beam/gaozhangyang/dataset/afdb_rep_mem.db \
    --data-splits '9990, 5, 5' \
    --precision="bf16" \
    --num-gpus 1 \
    --num-nodes 1 \
    --num-steps 100000 \
    --val-check-interval 5000 \
    --result-dir ./results/task_inversefolding/ \
    --min-seq-length 512 \
    --max-seq-length 512 \
    --resume-if-exists \
    --limit-val-batches 10 \
    --micro-batch-size 8 \
    --num-layers 12 \
    --hidden-size 480 \
    --num-attention-head 20 \
    --ffn-hidden-size 1920 \
    --tensor-model-parallel-size 1 \
    --create-tensorboard-logger \
    --wandb-offline 1 \
    --lr 1e-3 \
    --prefix_len 0 \
    --experiment-name ailab_IF_enc12_100k_nn64 \
    --log-every-n-steps 100 \
    --enc_layers 12 \
    --dec_layers 12 \
    --infer_feats 0 \
    --nn_neighbors 64