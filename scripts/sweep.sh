#!/bin/bash
#SBATCH --account=rrg-mbowling-ad
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

echo "Cloning repo..."
git config --global http.proxy 'socks5://127.0.0.1:8888'
git clone --quiet git@github.com:artemis79/PPO-implementation.git

curl -LsSf https://astral.sh/uv/install.sh | sh
cd PPO-implementation

uv venv $SLURM_TMPDIR/.venv --python 3.10
source $SLURM_TMPDIR/.venv/bin/activate

uv pip install -r requirements.txt --cache-dir $SLURM_TMPDIR/uv/cache
uv pip install gymnasium --cache-dir $SLURM_TMPDIR/uv/cache
uv pip install h5py --cache-dir $SLURM_TMPDIR/uv/cache
uv pip install requests[socks] 

if [[ "$1" == "new" ]]; then
   python wandb_sweep.py --sweep_id "new" --config "config.yaml"
elif [[ "$1" == "sweep_id" ]]; then
    for TIMES in {1..5}
    do
       python wandb_sweep.py --sweep_id "$2"
    done
   python wandb_sweep.py --sweep_id "$2"
else
   echo "Not a valid sweep"
fi
