#!/bin/bash
#SBATCH --account=def-mbowling
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=0-2:59
#SBATCH --cpu-freq=Performance
#SBATCH --array=1-50
#SBATCH --output=sweep_ppo/ppo_sweep_%j.out



module load python/3.10.13
if [ "$SLURM_TMPDIR" == "" ]; then
    exit 1
fi

cd $SLURM_TMPDIR


virtualenv pyenv
source pyenv/bin/activate

pip install 'requests[socks]' --no-index
pip install -r /home/mrahmani/scratch/requirements.txt


echo "Cloning repo..."
git config --global http.proxy 'socks5://127.0.0.1:8888'
git clone --quiet git@github.com:artemis79/PPO-implementation.git

cd PPO-implementation/

wandb agent university-alberta/ppo-tmp/qxspt3eu