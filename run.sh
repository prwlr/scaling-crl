#!/bin/sh

# If arg for path was passed
if [[ -n "$1" ]] ; then
  CHECKPOINT_ARG="--from_checkpoint_path $1"
else
  CHECKPOINT_ARG=""
fi

# Login to local wandb instance
# wandb login --host=http://localhost:8080

# NOTE: Take care that there are enough env_steps
# for envs to complete an episode each, each unroll
unroll_length=64 # Number of samples in-between optimisations
num_epochs=128 # Number of datapoints

# Limited to 256 @ 16GB
num_envs=256
num_eval_envs=128

# million=$((10**6))
bin_million=$((2**20)) # Approx. a million
# env_steps=$((100 * $million)) # About 2.3 hours @ 256 envs
# env_steps=$((10 * $million)) # About 17 minutes @ 256 envs
# env_steps=$((1 * $million)) # About 3 minutes @ 256 envs
env_steps=$((32 * $bin_million))

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
  --num_eval_envs $num_eval_envs \
  --total_env_steps $env_steps \
  --critic_depth 16 \
  --actor_depth 16 \
  --actor_skip_connections 4 \
  --critic_skip_connections 4 \
  --batch_size $batch_size \
  --unroll_length $unroll_length \
  --vis_length 1024 \
  --save_buffer 0 \
  $CHECKPOINT_ARG

# Symlink the lastest run in script directory
rm ./latest-run -r
ln -sf ./runs/$(ls ./runs/ | tail -1)/ ./latest-run

# Sync run with the local server
# wandb sync wandb/latest-run
