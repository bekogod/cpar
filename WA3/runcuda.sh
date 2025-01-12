#!/bin/bash
#SBATCH --partition=day
#SBATCH --constraint=k20
#SBATCH --ntasks=1
#SBATCH --time=5:00


nvprof ./bin/fluid_sim
