#!/bin/bash
#SBATCH --job-name=run_all_models                       # 📛 Master job name
#SBATCH --partition=gengpu                              # 🎯 GPU partition
#SBATCH --gres=gpu:1                                    # 🧠 One GPU to launch jobs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G                                        # 🧹 Minimal mem just for launching
#SBATCH --time=00:10:00                                 # ⏳ Short time limit to submit jobs
#SBATCH --output=$(pwd)/results/run_all_%j.out          # 📤 Master output
#SBATCH --error=$(pwd)/results/run_all_%j.err           # 📥 Master error

# ===== 🧠 SLURM self-submission =====
if [ -z "$SLURM_JOB_ID" ]; then
  echo "🟡 Not inside SLURM — submitting to Hyperion..."
  sbatch "$0"
  exit
fi

echo "✅ Running inside SLURM job ID: $SLURM_JOB_ID"

# ===== 🧩 Model Variants =====
MODELS=("gray2real_baseline" "gray2real" "cgan_initial" "cgan_final" "cgan_both" "attention")

# ===== ⚙️ Customizable Parameters (can override) =====
NITER=${NITER:-200}
NITER_DECAY=${NITER_DECAY:-100}
SAVE_EPOCH_FREQ=${SAVE_EPOCH_FREQ:-1}
PRINT_FREQ=${PRINT_FREQ:-50}
SAVE_LATEST_FREQ=${SAVE_LATEST_FREQ:-1000}
SAVE_BY_ITER=${SAVE_BY_ITER:-false}
GPU_ID=${GPU_ID:-0}
DATASET=${DATASET:-./datasets/cuhk}
DIRECTION=${DIRECTION:-gray2real}
OFFLINE=${OFFLINE:-false}

# ===== 📁 Log Folder Setup =====
mkdir -p results/logs

# ===== 🚀 Submit Training Jobs =====
for MODEL_TYPE in "${MODELS[@]}"; do
  EXPERIMENT_NAME="${MODEL_TYPE}_final"

  # 🧠 Check if model implementation exists
  MODEL_PATH="models/${MODEL_TYPE}_model.py"
  if [[ ! -f "$MODEL_PATH" ]]; then
    echo "⚠️ Skipping $MODEL_TYPE — model not implemented yet"
    continue
  fi

  echo "📤 Submitting training for $MODEL_TYPE"

  sbatch --job-name="$MODEL_TYPE" \
    --partition=gengpu \
    --gres=gpu:1 \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=4 \
    --mem=16G \
    --time=12:00:00 \
    --output="results/logs/${MODEL_TYPE}_%j.out" \
    --error="results/logs/${MODEL_TYPE}_%j.err" \
    --export=ALL,\
MODEL_TYPE=$MODEL_TYPE,\
EXPERIMENT_NAME=$EXPERIMENT_NAME,\
DATASET=$DATASET,\
DIRECTION=$DIRECTION,\
GPU_ID=$GPU_ID,\
NITER=$NITER,\
NITER_DECAY=$NITER_DECAY,\
SAVE_EPOCH_FREQ=$SAVE_EPOCH_FREQ,\
PRINT_FREQ=$PRINT_FREQ,\
SAVE_LATEST_FREQ=$SAVE_LATEST_FREQ,\
SAVE_BY_ITER=$SAVE_BY_ITER,\
OFFLINE=$OFFLINE \
    ./run_model.sh
done

echo "🚀 All training jobs submitted. Use 'squeue -u $USER' or tail logs to monitor."

echo ""
echo "📊 Summary of submitted jobs:"
for MODEL_TYPE in "${MODELS[@]}"; do
  echo "  ➤ $MODEL_TYPE → logs: results/logs/${MODEL_TYPE}_<JOBID>.out"
done

echo ""
echo "🛠️  To monitor training progress:"
echo "  • Check logs:      tail -f results/logs/*.out"
echo "  • Job queue:       squeue -u $USER"
echo "  • Finished jobs:   sacct -u $USER --format=JobID,JobName,State,Elapsed"
echo "  • WandB dashboard: https://wandb.ai"
echo ""
