#!/bin/sh

# If arg for path was passed
if [[ -n "$1" ]] ; then
  CHECKPOINT_ARG="--from_checkpoint_path $1"
else
  CHECKPOINT_ARG=""
fi

# Login to local wandb instance
# wandb login --host=http://localhost:8080

# It should hold that:
# env_steps / (num_epochs * num_envs) >= episode_length
num_epochs=16
num_envs=128

# million=$((10**6))
bin_million=$((2**20)) # Ca. a million
# env_steps=$((100 * $million)) # About 2.3 hours @ 256 envs
# env_steps=$((10 * $million)) # About 17 minutes @ 256 envs
# env_steps=$((1 * $million)) # About 3 minutes @ 256 envs
env_steps=$((2 * $bin_million))

# batch_size=8192 # Doable at depth 16 with 256 envs
batch_size=512

# Run the training
python train.py \
  --project_name "scaling-crl" \
  --env_id "humanoid" \
  --eval_env_id "humanoid" \
  --episode_length 1024 \
  --num_epochs $num_epochs \
  --num_envs $num_envs \
  --total_env_steps $env_steps \
  --critic_depth 16 \
  --actor_depth 16 \
  --actor_skip_connections 4 \
  --critic_skip_connections 4 \
  --batch_size $batch_size \
  --vis_length 1024 \
  --save_buffer 0 \
  $CHECKPOINT_ARG

# Symlink the lastest run in script directory
rm ./latest-run -r
ln -sf ./runs/$(ls ./runs/ | tail -1)/ ./latest-run

# Sync run with the local server
# wandb sync wandb/latest-run
