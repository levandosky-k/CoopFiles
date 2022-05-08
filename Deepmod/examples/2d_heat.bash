#!/bin/bash
#BATCH --partition=short
#SBATCH --job-name=test
#SBATCH --time=10:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --output=%j.output
#SBATCH --error=%j.error
ipython Diffusion2D.ipynb
