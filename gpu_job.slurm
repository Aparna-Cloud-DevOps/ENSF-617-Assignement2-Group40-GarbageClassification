#!/bin/bash
#SBATCH --job-name=garbage_train                 # Name of the job
#SBATCH --partition=gpu                          # Request GPU partition
#SBATCH --gres=gpu:1                              # Request 1 GPU
#SBATCH --cpus-per-task=1                         # Number of CPU cores
#SBATCH --mem=8G                                  # Memory allocation
#SBATCH --time=02:00:00                           # Max runtime (HH:MM:SS)
#SBATCH --output=garbage_test_efficientnet_%j.out # STDOUT log file (%j = job ID)
#SBATCH --error=garbage_test_efficientnet__%j.err # STDERR log file

# ------------------------------------------------------------
# SAFETY: Exit immediately if any command fails
# ------------------------------------------------------------
set -e

# ------------------------------------------------------------
# LOAD CUDA MODULE (matches your cluster environment)
# ------------------------------------------------------------
module load cuda/12.6.2

# ------------------------------------------------------------
# ACTIVATE CONDA ENVIRONMENT
# ------------------------------------------------------------
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gpuenv

# ------------------------------------------------------------
# NAVIGATE TO PROJECT DIRECTORY
# ------------------------------------------------------------
echo "Changing directory..."
cd /home/aparna.ayyalasomayaj/garbage_classification || { 
    echo "ERROR: Failed to change directory"; 
    exit 1; 
}

# ------------------------------------------------------------
# RUN PYTHON TEST SCRIPT
# ------------------------------------------------------------
echo "Starting EfficientNet test job..."
python -u test_efficientnet.py     # -u ensures unbuffered output (real-time logs)
echo "Job completed successfully."
