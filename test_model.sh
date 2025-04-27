#!/bin/bash
#SBATCH --job-name=model_test                          # 🧪 Test job name
#SBATCH --partition=preemptgpu
#SBATCH --gres=gpu:1                                   # 🧠 Request 1 GPU
#SBATCH --nodes=1                                      # 🖥️ Run on 1 node
#SBATCH --ntasks=1                                     # 🚶 Run 1 task
#SBATCH --cpus-per-task=4                              # 🧵 Use 4 CPU cores
#SBATCH --mem=16G                                      # 💾 Memory
#SBATCH --time=01:00:00                                # ⏳ 1 hour time limit
#SBATCH --output=results/%x_%j.out                     # 📤 Master output
#SBATCH --error=results/%x_%j.err                      # 📥 Master error

# ===== 🧩 Configurable Parameters =====
MODEL_TYPE=${MODEL_TYPE:-gray2real}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-gray2real_final}
DATASET=${DATASET:-./datasets/cuhk}
DIRECTION=${DIRECTION:-gray2real}
GPU_ID=${GPU_ID:-0}
NUM_TEST=${NUM_TEST:-50}
EPOCH=${EPOCH:-latest}
CHECKPOINTS_DIR=${CHECKPOINTS_DIR:-./checkpoints}


# 🛑 Validate MODEL_TYPE
VALID_MODELS=("gray2real_baseline" "gray2real" "cgan_initial" "cgan_final" "cgan_both" "sketch2face" "attention")
if [[ ! " ${VALID_MODELS[@]} " =~ " ${MODEL_TYPE} " ]]; then
  echo "❌ Error: Unsupported MODEL_TYPE '$MODEL_TYPE'"
  echo "🧪 Valid options: ${VALID_MODELS[*]}"
  exit 1
fi

# 🔍 SLURM check
if [ -z "$SLURM_JOB_ID" ]; then
  echo "🟡 Not inside SLURM — requesting GPU node via srun..."
  srun --partition=preemptgpu --gres=gpu:1 --mem=16G --cpus-per-task=4 --time=01:00:00 --pty "$0"
  exit
fi

echo "✅ Running inside SLURM job ID: $SLURM_JOB_ID"
mkdir -p results

# ===== 🔧 Environment Setup =====
source /opt/flight/etc/setup.sh
flight env activate gridware
module load libs/nvidia-cuda/11.2.0/bin
module load gnu
source gray_env/bin/activate

export https_proxy=http://hpc-proxy00.city.ac.uk:3128
export http_proxy=http://hpc-proxy00.city.ac.uk:3128
export TORCH_HOME=/mnt/data/public/torch

echo "🐍 Python version:"
python --version
echo "🖥️ GPU info:"
nvidia-smi

# 🌐 Login to WandB
export WANDB_API_KEY=f1b1dcb5ebf893b856630d4481b7d7cd05101b45
echo "🔐 Logging into WandB..."
wandb login $WANDB_API_KEY --relogin


# ===== 🧪 Run Testing =====
echo "🧠 Model Type      : $MODEL_TYPE"
echo "📁 Dataset         : $DATASET"
echo "📂 Experiment Name : $EXPERIMENT_NAME"
echo "➡️  Direction       : $DIRECTION"
echo "🧪 Num Test Images : $NUM_TEST"

EXTRA_FLAGS=""

python3 test.py \
  --dataroot "$DATASET" \
  --name "$EXPERIMENT_NAME" \
  --model "$MODEL_TYPE" \
  --dataset_mode cuhk \
  --direction "$DIRECTION" \
  --input_nc 1 \
  --output_nc 3 \
  --gpu_ids "$GPU_ID" \
  --num_test "$NUM_TEST" \
  --checkpoints_dir "$CHECKPOINTS_DIR"\
  --epoch "$EPOCH"\
  --eval \
$EXTRA_FLAGS


echo "✅ Testing complete for $MODEL_TYPE 🎉"
