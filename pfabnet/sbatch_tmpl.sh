#!/bin/bash -l
#SBATCH -e %j.err
#SBATCH -o %j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=32gb
#SBATCH --wait
