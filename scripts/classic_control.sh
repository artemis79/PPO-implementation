#!/bin/bash
#SBATCH --account=def-mbowling
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=0-0:59
#SBATCH --cpu-freq=Performance
#SBATCH --array=1-50
#SBATCH --output=outputs_ppo/ppo_%j.out



module load python/3.10.13
if [ "$SLURM_TMPDIR" == "" ]; then
    exit 1
fi

cd $SLURM_TMPDIR


virtualenv pyenv
source pyenv/bin/activate

pip install 'requests[socks]' --no-index
pip install -r /home/mrahmani/requirements.txt --no-index


echo "Cloning repo..."
git config --global http.proxy 'socks5://127.0.0.1:8888'
git clone --quiet git@github.com:artemis79/PPO-implementation.git


python3 main_ppo.py --gym-id "MountainCar-v0" --track --seed $SLURM_ARRAY_TASK_ID --cuda False --total-timesteps 500000 --wandb-project-name "energy-ppo" --aggregate-function "mean"