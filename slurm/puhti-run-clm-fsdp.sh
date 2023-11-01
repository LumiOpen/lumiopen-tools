#!/bin/bash

#SBATCH --account=project_2007628
#SBATCH --partition=gputest
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive=user
#SBATCH --time=0:15:00
#SBATCH --mem=0
#SBATCH --gres=gpu:v100:4
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# symlink logs/latest.out and logs/latest.err
ln -f -s $SLURM_JOB_ID.out logs/latest.out
ln -f -s $SLURM_JOB_ID.err logs/latest.err

set -euo pipefail

module load pytorch/2.0
export SING_IMAGE=/appl/soft/ai/singularity/images/pytorch_2.0.1_csc_mlflow_static_fix.sif

export TRANSFORMERS_CACHE=/scratch/project_2007628/transformers_cache

MODEL="TurkuNLP/gpt3-finnish-8B"

export RDZV_HOST=$(hostname)
export RDZV_PORT=29400
export RDZV_ID="$SLURM_JOB_ID"
export RDZV_ENDPOINT="$RDZV_HOST:$RDZV_PORT"

echo "rdzv endpoint: $RDZV_ENDPOINT"

srun python3 -m torch.distributed.run \
       --nnodes=$SLURM_JOB_NUM_NODES \
       --nproc_per_node 4 \
       --rdzv_backend=c10d \
       --rdzv_id="$RDZV_ID" \
       --rdzv_endpoint="$RDZV_ENDPOINT" \
       run_clm.py \
       --model_name_or_path "$MODEL" \
	--output_dir output \
	--overwrite_output_dir \
	--dataset_name wikitext \
	--dataset_config wikitext-2-v1 \
	--do_train \
	--per_device_train_batch_size 1 \
	--per_device_eval_batch_size 1 \
	--num_train_epochs 1 \
	--max_train_samples 100 \
	--fp16 \
	--fp16_full_eval \
	--block_size 1024 \
	--fsdp "full_shard auto_wrap"

echo "done."
