#!/bin/sh

python train.py \
  --env_id "humanoid" \
  --eval_env_id "humanoid" \
  --num_epochs 10 \
  --num_envs 2 \
  --total_env_steps 100000 \
  --critic_depth 2 \
  --actor_depth 2 \
  --actor_skip_connections 1 \
  --critic_skip_connections 1 \
  --batch_size 32 \
  --vis_length 10 \
  --save_buffer 0
