#!/bin/bash
#SBATCH --job-name=leo
#SBATCH --time=0-0:5
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --output=saida%j.out

./pca /datasets/125.dat 1 20
