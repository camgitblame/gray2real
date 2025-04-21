#!/bin/bash

# === 1. ENV SETUP ===
echo "🔧 Loading modules and setting up environment..."
# 📦 Activate gridware environment
source /opt/flight/etc/setup.sh
flight env activate gridware

# 🚀 Load CUDA & compiler modules
module load gnu
module load compilers/gcc
module load libs/nvidia-cuda/11.2.0/bin  

# === 2. PROXY CONFIG ===
export https_proxy=http://hpc-proxy00.city.ac.uk:3128
export http_proxy=http://hpc-proxy00.city.ac.uk:3128

# 🧠 Set Torch cache dir
export TORCH_HOME=/mnt/data/public/torch

# === 3. INSTALL PYENV IF MISSING ===
if [ ! -d "$HOME/.pyenv" ]; then
    echo "📦 Installing pyenv and pyenv-virtualenv..."
    git config --global http.proxy $https_proxy
    git clone https://github.com/pyenv/pyenv.git ~/.pyenv
    git clone https://github.com/pyenv/pyenv-virtualenv.git ~/.pyenv/plugins/pyenv-virtualenv
else
    echo "✅ pyenv already installed."
fi

# 📂 Load pyenv into this shell
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

# 🧪 Check System Info
echo "🐍 Python version (pre-install):"
python --version || echo "Python not found"
echo "🖥️ GPU info:"
nvidia-smi

# === 4. INSTALL PYTHON 3.9.5 IF MISSING ===
if ! pyenv versions --bare | grep -q "3.9.5"; then
    echo "🐍 Installing Python 3.9.5 with proxy-aware compile flags..."
    CPPFLAGS="-I/opt/apps/gnu/include" \
    LDFLAGS="-L/opt/apps/gnu/lib -L/opt/apps/gnu/lib64 -ltinfo" \
    pyenv install 3.9.5
else
    echo "✅ Python 3.9.5 already installed."
fi

pyenv global 3.9.5

# === 5. CREATE VIRTUALENV IF MISSING ===
if ! pyenv virtualenvs --bare | grep -q "gray_env"; then
    echo "📁 Creating virtualenv: gray_env..."
    pyenv virtualenv 3.9.5 gray_env
else
    echo "✅ Virtualenv 'gray_env' already exists."
fi

# === 6. LINK ENV TO PROJECT FOLDER ===
mkdir -p gray2real
echo "gray_env" > gray2real/.python-version

# === 7. ACTIVATE VENV AND INSTALL DEPENDENCIES ===
cd gray2real
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
pyenv activate gray_env

echo "🐍 Python version (inside virtualenv):"
python --version

# === 8. INSTALL REQUIREMENTS ===
if [ -f requirements.txt ]; then
    echo "📦 Installing Python packages from requirements.txt..."
    pip install --proxy $https_proxy -r requirements.txt
else
    echo "⚠️ No requirements.txt found — skipping dependency install."
fi

# === 9. INSTALL PYTORCH (CUDA) ===
echo "🔥 Installing PyTorch with CUDA support (CUDA 11.8)..."
pip install --proxy $https_proxy torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo "✅ All setup complete! Virtualenv 'gray_env' is ready inside gray2real/ 🎉"
