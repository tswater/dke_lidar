#!/bin/sh
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --job-name="les_qik"
#SBATCH --output="log.txt"
#SBATCH --exclusive
#SBATCH --exclude=node5,node6,node7,node8
time python process_les.py

