#!/bin/bash
#SBATCH --job-name=tta-cotta-commonvoice-6000            # Job name
#SBATCH --output=tta-cotta-hubert-commonvoice-6000.out          # Output file name
#SBATCH --error=tta-cotta-hubert-commonvoice-6000.err           # Error file name
#SBATCH --time=24:00:00               # Maximum runtime (24 hours)
#SBATCH -p gpu-a100                   # Partition name 
#SBATCH --gres=gpu:1                  # Number of GPUs 
#SBATCH --cpus-per-task=8             # Number of CPU cores per task
#SBATCH --ntasks=1                    # Number of tasks (processes)

# Load necessary modules
module load GCC/11.3.0
module load virtualenv/20.23.1-Python-3.11.3
module load CUDA/12.5.1
module load Python/3.11.3
# Activate your virtual environment
source ~/sutaenv/bin/activate
pip install cma
pip install audiomentations
# Run your Python script
time srun python cotta.py --asr facebook/hubert-large-ls960-ft \
                --steps 1 \
                --batch_size 1 \
                --dataset_name commonvoice \
                --dataset_dir /home/jiahengd/tta-suta/cv-corpus-5.1-2020-06-22/en \
                --split test \
                --extra_noise 0.0 \
                --lr 2e-5

deactivate

my-job-stats -a -n -s