#!/bin/bash
#SBATCH --job-name=model_train                         # 📛 Job name
#SBATCH --partition=gengpu                              # 🎯 GPU partition on Hyperion
#SBATCH --gres=gpu:1                                    # 🧠 Request 1 GPU
#SBATCH --nodes=1                                       # 🖥️ 1 node
#SBATCH --ntasks=1                                      # 🚶 1 task
#SBATCH --cpus-per-task=4                               # 🧵 4 CPU threads
#SBATCH --mem=16G                                       # 💾 Memory
#SBATCH --time=02:00:00                                 # ⏳ 2h time limit
#SBATCH --output=$(pwd)/results/%x_%j.out               # 📤 Stdout
#SBATCH --error=$(pwd)/results/%x_%j.err                # 📥 Stderr
                     

# ===== 🧩 Configurable Parameters =====
MODEL_TYPE=${MODEL_TYPE:-gray2real}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-gray2real_experiment}
DATASET=${DATASET:-./datasets/cuhk}
DIRECTION=${DIRECTION:-gray2real}
GPU_ID=${GPU_ID:-0}

# 🛑 Validate MODEL_TYPE
VALID_MODELS=("gray2real_baseline" "gray2real" "cgan_initial" "cgan_final" "cgan_both" "attention")
if [[ ! " ${VALID_MODELS[@]} " =~ " ${MODEL_TYPE} " ]]; then
  echo "❌ Error: Unsupported MODEL_TYPE '$MODEL_TYPE'"
  echo "🧪 Valid options: ${VALID_MODELS[*]}"
  exit 1
fi

# 🔍 Check if we're already inside a SLURM job
if [ -z "$SLURM_JOB_ID" ]; then
  echo "🟡 Not inside SLURM — requesting GPU node via srun..."
  srun --partition=gengpu --gres=gpu:1 --mem=16G --cpus-per-task=4 --time=02:00:00 --pty "$0"
  exit
fi

echo "✅ Running inside SLURM job ID: $SLURM_JOB_ID"

# ===== 📂 Check if results folder exists ===== 
mkdir -p results

# ===== 🔧 Environment Setup =====

# 📦 Activate gridware environment
source /opt/flight/etc/setup.sh
flight env activate gridware

# 🚀 Load CUDA & compiler modules
module load libs/nvidia-cuda/11.2.0/bin
module load gnu

# Activate virtual environment
source gray_env/bin/activate

# 🌐 Export proxy (required for WandB or pip install)
export https_proxy=http://hpc-proxy00.city.ac.uk:3128
export http_proxy=http://hpc-proxy00.city.ac.uk:3128

# Set Torch cache dir
export TORCH_HOME=/mnt/data/public/torch

# Check software versions
echo "🐍 Python version:"
python --version
echo "🖥️ GPU info:"
nvidia-smi

# Login to WandB (optional)
export WANDB_API_KEY=f1b1dcb5ebf893b856630d4481b7d7cd05101b45
echo "🔐 Logging into WandB..."
wandb login $WANDB_API_KEY --relogin

# ===== Run Training =====
echo "🧠 Model Type      : $MODEL_TYPE"
echo "📁 Dataset         : $DATASET"
echo "📂 Experiment Name : $EXPERIMENT_NAME"
echo "➡️  Direction       : $DIRECTION"
echo "🎨 Input Channels  : 1"
echo "🌈 Output Channels : 3"


python3 train.py \
  --dataroot "$DATASET" \
  --name "$EXPERIMENT_NAME" \
  --model "$MODEL_TYPE" \
  --dataset_mode cuhk \
  --direction "$DIRECTION" \
  --input_nc 1 \
  --output_nc 3 \
  --batch_size 1 \
  --niter 10 \
  --niter_decay 10 \
  --lambda_L1 100 \
  --gpu_ids "$GPU_ID" \
  --save_by_iter \
  --verbose 

echo "✅ Training complete for $MODEL_TYPE 🎉"
