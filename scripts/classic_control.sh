#!/bin/bash

for SEED in {1..50}
do
    python3 main_ppo.py --gym-id "MountainCar-v0" --track --seed $SEED --cuda False --total-timesteps 500000 --wandb-project-name "ppo-count"
done