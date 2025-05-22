#!/bin/bash

python mappo.py \
  --map_file map1.txt \
  --n_robots 2 \
  --n_packages 5 \
  --max_time_steps 100 \
  --algorithm MAPPO \
  --max_train_steps 5000000 \
  --evaluate_freq 20000 \
  --evaluate_times 32 \
  --save_freq 100000 \
  --batch_size 32 \
  --mini_batch_size 8 \
  --rnn_hidden_dim 128 \
  --mlp_hidden_dim 128 \
  --lr 0.0005 \
  --gamma 0.99 \
  --lamda 0.95 \
  --epsilon 0.2 \
  --K_epochs 15 \
  --entropy_coef 0.01 \
  --use_adv_norm \
  --use_reward_norm \
  --use_lr_decay \
  --use_grad_clip \
  --use_orthogonal_init \
  --set_adam_eps \
  --use_relu \
  --use_rnn \
  --use_agent_specific \
  --add_agent_id
  # --use_value_clip

  # --use_reward_scaling \