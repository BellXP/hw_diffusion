#!/usr/bin/bash

cd /nvme/xupeng/workplace/hw_diffusion
pwd

JOB_NAME=hw_diffusion
NOW="`date +%Y%m%d%H%M%S`"
output_dir='/nvme/xupeng/workplace/exp_dir'
log_dir='/nvme/xupeng/workplace/logs'
PYTHON=/nvme/xupeng/miniconda3/envs/bell/bin/python

export CUDA_VISIBLE_DEVICES=2

# --use-checkpoint
# --use-vae
# --perform-lcp
# --use-condition

$PYTHON main.py \
--exp-name $JOB_NAME \
--output $output_dir \
--model-epochs 50 \
--predictor-epochs 50 \
--train-batch-size 256 \
--val-batch-size 1024 \
--use-checkpoint --use-vae \
2>&1 | tee $log_dir/${JOB_NAME}_${NOW}.log