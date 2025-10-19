#!/bin/bash
#SBATCH --job-name=E-BATS-h-str-simu-4.3.0    # Job name
#SBATCH --output=E-BATS-h-str-simu-4.3.0.out         # Output file name
#SBATCH --error=E-BATS-h-str-simu-4.3.0.err           # Error file name
#SBATCH --time=8:00:00               # Maximum runtime (24 hours)
#SBATCH -p gpu-a100                  # Partition name 
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
pip install cma==4.3.0
pip install matplotlib
# facebook/wav2vec2-base-960h (ema_decay=0.9), facebook/hubert-large-ls960-ft (ema_decay=0.8)
time srun python E-BATS.py --asr facebook/hubert-large-ls960-ft \
                --steps 25 \
                --pop_size 50 \
                --batch_size 1 \
                --dataset_name chime \
                --early_stop_threshold 0.001\
                --patient 3\
                --dataset_dir /home/jiahengd/tta-suta/CHiME3 \
                --extra_noise 0.0 \
                --beta 2.0\
                --alpha 1.0\
                --temp 2.0\
                --split str-simu \
                --reset_frequency 1 \
                --ema_decay 0.8 \
                --use_tema \
                --covariance_ratio 1.0 \
                --random_seed 2024 \
                --confidence_max 5.0 \
                --tokenwise \
                

deactivate

my-job-stats -a -n -s