#!/bin/bash
#SBATCH --job-name=leo
#SBATCH --time=0-0:1
#SBATCH --partition=test
#SBATCH --output=saida%j.sc

./pca ./datasets/125.dat 1 32
./pca ./datasets/125.dat 2 32
./pca ./datasets/125.dat 4 32
./pca ./datasets/125.dat 8 32
./pca ./datasets/125.dat 16 32
./pca ./datasets/125.dat 32 32
