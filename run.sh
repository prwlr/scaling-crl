#!/bin/sh

# Login to local wandb instance
# wandb login --host=http://localhost:8080

million=$((10**6))

# env_steps=$((100 * $million)) # About 21 hours @ 1 env
env_steps=$((10 * $million)) # About 17 minutes @ 256 envs
# env_steps=$((1 * $million)) # About 3 minutes @ 256 envs

# batch_size=2048 # Doable at depth 16 with 256 envs
batch_size=2048

# Run the training
python train.py \
  --env_id "humanoid" \
  --eval_env_id "humanoid" \
  --num_epochs 100 \
  --num_envs 256 \
  --total_env_steps $env_steps \
  --critic_depth 16 \
  --actor_depth 16 \
  --actor_skip_connections 4 \
  --critic_skip_connections 4 \
  --batch_size $batch_size \
  --vis_length 1000 \
  --save_buffer 0

# Sync run with the local server
# wandb sync wandb/latest-run
