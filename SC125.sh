#!/bin/bash
#SBATCH --job-name=leo
#SBATCH --time=0-0:5
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --output=saida%j.out

./pca ./datasets/125.dat 1 64
./pca ./datasets/125.dat 2 64
./pca ./datasets/125.dat 4 64
./pca ./datasets/125.dat 8 64
./pca ./datasets/125.dat 16 64
./pca ./datasets/125.dat 32 64
