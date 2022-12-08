#!/bin/bash

#SBATCH --job-name yanfu_595_proj
#SBATCH --partition=spgpu
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-cpu=16GB
#SBATCH --time=01:00:00
#SBATCH --account=eecs595f22_class
#SBATCH --mail-type=BEGIN,END,FAIL


# set up job

pushd /home/yanfuguo/proj/
module load python/3.9 cuda
# source env/bin/activate

pip3 list
which python3

# run job
python3 model_evaluate_task2.py