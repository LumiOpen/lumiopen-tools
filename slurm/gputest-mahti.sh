#!/bin/bash

# Start interactive session on a GPU node on the Mahti supercomputer.
# (You'll probably need to edit the --account parameter to use this.)

srun \
    --account=project_2007628 \
    --partition=gputest \
    --ntasks=1 \
    --cpus-per-task=32 \
    --gres=gpu:a100:4 \
    --time=0:15:00 \
    --mem=128G \
    --pty \
    bash
