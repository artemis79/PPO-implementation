#!/bin/bash

for SEED in {1..5}
do
    python3 main_ppo.py --gym-id CartPole-v1 --track --seed $SEED --cuda False --total-timesteps 50000
done