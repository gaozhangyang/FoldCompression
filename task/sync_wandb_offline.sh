# 1) 登录（只需一次）
wandb login   # 粘贴你的 API key

# ddb1831ecbd2bf95c3323502ae17df6e1df44ec0

# 2) 同步到指定项目/团队（entity 可选）
wandb sync \
  --project foldtoken5 \
  --entity gaozhangyang \
  ./offline-run-20250919_120636-c6m1ppw5
