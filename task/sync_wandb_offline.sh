# 1) 登录（只需一次）
wandb login   # 粘贴你的 API key

# ddb1831ecbd2bf95c3323502ae17df6e1df44ec0

# 2) 同步到指定项目/团队（entity 可选）
wandb sync \
  --project foldtoken5 \
  --entity gaozhangyang \
  /mnt/shared-storage-user/gaozhangyang/workspace/FoldCompression/results/struct_compress/eval_nn10_in_str_out_str_contrastive_lr1e4/wandb/offline-run-20251016_160447-hf148u1n
