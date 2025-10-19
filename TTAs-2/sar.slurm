#!/bin/bash
#SBATCH --job-name=tta-sar          # Job name
#SBATCH --output=tta-sar-commonvoice=filter.out          # Output file name
#SBATCH --error=tta-sar-commonvoice=filter.err           # Error file name
#SBATCH --time=4:00:00               # Maximum runtime (24 hours)
#SBATCH -p gpu-a100-short                  # Partition name 
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
#facebook/wav2vec2-base-960h
#facebook/hubert-large-ls960-ft
#/home/jiahengd/tta-suta/TEDLIUM_release2/test
#/home/jiahengd/tta-suta/cv-corpus-5.1-2020-06-22/en
# Run your Python script
time srun python SAR.py --asr facebook/wav2vec2-base-960h \
                --steps 1 \
                --batch_size 1 \
                --dataset_name commonvoice \
                --dataset_dir /home/jiahengd/tta-suta/cv-corpus-5.1-2020-06-22/en \
                --em_coef 1 \
                --log_dir ~/tta-suta/results/SAR/commonvoice \
                --lr 2e-5 \
                --non_blank \
                --split test \
                --extra_noise 0.0 \

deactivate

my-job-stats -a -n -s