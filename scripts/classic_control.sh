#!/bin/bash

for SEED in {1..50}
do
    python3 main_ppo_count.py --gym-id "CartPole-v1" --track --seed $SEED --cuda False --total-timesteps 500000 --wandb-project-name "ppo-count"
done