#!/bin/bash
echo $CUDA_VISIBLE_DEVICES
echo $SLURM_NODELIST
echo $SLURM_NODEID
ml purge
sbatch --gres=gpu:1 --time=60:00:00 --job-name=3032 diffuse1.sh 31800 32000
sbatch --gres=gpu:1 --time=60:00:00 --job-name=3234 diffuse1.sh 33680 34000
sbatch --gres=gpu:1 --time=60:00:00 --job-name=3436 diffuse1.sh 35835 36000
sbatch --gres=gpu:1 --time=60:00:00 --job-name=3638 diffuse1.sh 37990 38000