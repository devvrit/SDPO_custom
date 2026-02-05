#!/bin/bash

# Helper script to execute commands inside the Singularity container
# Usage: ./container_exec.sh <command>
# Example: ./container_exec.sh python data/load_dataset.py --dataset_name Physics
#
# Note: Singularity shares host network by default (no --network flag needed)

singularity exec --rocm \
  --env PYTHONDONTWRITEBYTECODE=1 \
  --env PYTHONUNBUFFERED=1 \
  --env PYTHONPATH=$PWD:$WORK/python_packages \
  -B $HOME:$HOME \
  -B $WORK:$WORK \
  --pwd $PWD \
  $WORK/rocm_sdpo.sif \
  "$@"
