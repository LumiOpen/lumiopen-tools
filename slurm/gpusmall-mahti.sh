#!/bin/bash

# Start interactive session on a GPU node on the Mahti supercomputer.
# (You'll probably need to edit the --account parameter to use this.)

srun \
    --account=project_2007628 \
    --partition=gpusmall \
    --ntasks=1 \
    --cpus-per-task=32 \
    --gres=gpu:a100:2 \
    --time=1:00:00 \
    --mem=128G \
    --pty \
    bash
