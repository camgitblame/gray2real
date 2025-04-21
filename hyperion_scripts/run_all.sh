#!/bin/bash

MODELS=("baseline" "gray2real" "cgan_initial" "cgan_final" "cgan_both" "sketch2face" "attention")

for MODEL in "${MODELS[@]}"; do
  EXPERIMENT="cuhk_${MODEL}_$(date +%Y%m%d_%H%M)"
  echo "📦 Queuing model: $MODEL → $EXPERIMENT"
  MODEL_TYPE=$MODEL EXPERIMENT_NAME=$EXPERIMENT DIRECTION=sketch2face sbatch run_model.sh
  sleep 1  # give SLURM a moment between jobs
done
