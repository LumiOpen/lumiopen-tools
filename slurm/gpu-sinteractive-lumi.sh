#!/bin/bash

# Start interactive session on a GPU node on the LUMI supercomputer.
# (You'll probably need to edit the --account parameter to use this.)

srun \
    --account=project_462000319 \
    --partition=dev-g \
    --ntasks=1 \
    --gres=gpu:mi250:8 \
    --time=1:00:00 \
    --mem=0 \
    --pty \
    bash
