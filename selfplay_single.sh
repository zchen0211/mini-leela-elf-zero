#!/bin/bash

#SBATCH --job-name=minigo-selfplay
#SBATCH --output=/checkpoint/%u/jobs/sample-%j.out
#SBATCH --error=/checkpoint/%u/jobs/sample-%j.err
#SBATCH --partition=learnfair

#SBATCH --nodes=1
#SBATCH --gres:gpu=1

### Section 2: Setting environment variables for the job

srun --label python local_rl_loop_9x9.py
