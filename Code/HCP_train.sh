#!/bin/sh
  
#SBATCH --nodes=1
  
# GPU
# SBATCH --gres=gpu:1
#SBATCH --nodelist=cn5  
#SBATCH --partition=centos7   
#SBATCH --ntasks=4
#SBATCH --mem=10000
#SBATCH --time=00:10:00
#SBATCH --job-name=GNN
# SBATCH --constraint=bigGPUmem
  

# python in virtual env
python HCP_train.py
