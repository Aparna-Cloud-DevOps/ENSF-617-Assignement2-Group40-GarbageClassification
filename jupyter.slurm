#!/bin/bash
#SBATCH --job-name=jupyter
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=jupyter_%j.out

module load cuda/12.6.2  # or whatever TALC uses
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gpuenv

# Launch Jupyter
jupyter notebook --no-browser --ip=0.0.0.0 --port=8890

