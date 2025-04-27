# gray2real Pipeline

This repo contains a training and testing setup for all models from the gray2real projects.


---

## 🛠️ Environment Setup Instructions

To prepare the full environment for training and testing models in this project, run the provided setup script:

1. **Go to gray2real project directory:**

```bash
cd gray2real/
```

2. Make sure the setup script is executable (This is only needed the first time)
```bash
chmod +x setup.sh
```

3. Run the setup script:
```bash
./setup.sh
```

## 🔥 What this script does:

| Step | What It Does | Details |
|:----|:-------------|:--------|
| 1 | Loads modules | Loads necessary compilers, CUDA 11.2 |
| 2 | Sets up proxies | Configures HTTP/HTTPS proxy if running behind a firewall (City University proxy) |
| 3 | Installs `pyenv` and `pyenv-virtualenv` | Only if they are missing |
| 4 | Installs Python 3.9.5 | From source, with proxy-aware build settings |
| 5 | Creates a clean virtualenv `gray_env` | Deletes any existing `gray_env` if found |
| 6 | Activates the new virtualenv | Ensures the virtual environment is properly used |
| 7 | Upgrades `pip`, `setuptools`, `wheel` | Keeps packaging tools up-to-date |
| 8 | Installs Python dependencies | Installs all packages from `requirements.txt` |
| 9 | Installs PyTorch with CUDA 11.8 | Ensures GPU acceleration is set up |
| 10 | Final checks | Shows Python version, lists installed torch and wandb versions |


## 🚀 Usage

### ✅ Train a model via SLURM

1. Baseline

```bash
MODEL_TYPE=gray2real_baseline \
EXPERIMENT_NAME=gray2real_baseline \
NITER=200 \
NITER_DECAY=100 \
SAVE_EPOCH_FREQ=1 \
PRINT_FREQ=50 \
SAVE_BY_ITER=false \
GPU_ID=0 \
./run_model.sh
```

2. gray2real with standard L1 and cGAN

```bash
MODEL_TYPE=gray2real \
EXPERIMENT_NAME=gray2real_final \
NITER=200 \
NITER_DECAY=100 \
SAVE_EPOCH_FREQ=1 \
PRINT_FREQ=50 \
SAVE_BY_ITER=false \
GPU_ID=0 \
./run_model.sh

```
3. gray2real with VGG loss

```bash
MODEL_TYPE=gray2real \
EXPERIMENT_NAME=gray2real_final_vgg \
NITER=30 \
NITER_DECAY=30 \
LOSS_MODE=vgg \
LAMBDA_L1=100 \
LAMBDA_VGG=10 \
PRINT_FREQ=50 \
SAVE_EPOCH_FREQ=5 \
SAVE_LATEST_FREQ=1000 \
SAVE_BY_ITER=false \
GPU_ID=0 \
./run_model.sh

```

4. gray2real with FaceNet Identity loss


```bash
MODEL_TYPE=gray2real \                                             
EXPERIMENT_NAME=gray2real_final_id \
NITER=30 \
NITER_DECAY=30 \
LOSS_MODE=id \
LAMBDA_ID=5 \
GPU_ID=0 \
./run_model.sh

```

4. gray2real with combined VGG and FaceNet Identity loss

```bash

MODEL_TYPE=gray2real \
EXPERIMENT_NAME=gray2real_final_both \
NITER=30 \
NITER_DECAY=30 \
LOSS_MODE=both \
LAMBDA_L1=100 \
LAMBDA_VGG=10 \
LAMBDA_ID=5 \
PRINT_FREQ=50 \
SAVE_EPOCH_FREQ=5 \
SAVE_LATEST_FREQ=1000 \
SAVE_BY_ITER=false \
GPU_ID=0 \
./run_model.sh
```


---

## ⚙️ Customizable Parameters for Training

When using `run_model.sh` or `run_all.sh`, you can override the default training settings by passing environment variables. Here's a full list of customizable options:

| Variable            | Description                                                       | Example                         |
|---------------------|-------------------------------------------------------------------|---------------------------------|
| `MODEL_TYPE`        | Model variant to train (must match implemented class name)       | `gray2real`, `gray2real_baseline`        |
| `EXPERIMENT_NAME`   | Name used for saving logs and checkpoints                        | `gray2real_final`               |
| `DATASET`           | Path to the training dataset                                      | `./datasets/cuhk`               |
| `DIRECTION`         | Translation direction (e.g., `gray2real`, `AtoB`)                 | `gray2real`                     |
| `GPU_ID`            | Which GPU ID to use (note: SLURM assigns GPU via `--gres=gpu:1`) | `0`                             |
| `NITER`             | Number of epochs with constant learning rate                     | `200`                           |
| `NITER_DECAY`       | Number of epochs where learning rate linearly decays             | `100`                           |
| `SAVE_EPOCH_FREQ`   | Save checkpoint every N epochs                                    | `1`                             |
| `SAVE_LATEST_FREQ`  | Save the latest checkpoint every N iterations                    | `1000`                          |
| `PRINT_FREQ`        | Log training progress every N iterations                         | `50`                            |
| `SAVE_BY_ITER`      | Save by iteration number instead of by epoch (true/false)        | `false`                         |
| `OFFLINE`           | Set to `true` to log to WandB offline, `false` for online        | `false`                         |



## 🛠️ Advanced

### ✅ Train all models at once

```bash
bash ./run_all.sh
```

## 📡 Monitoring Training Jobs on Hyperion

If you've launched jobs using `run_all.sh`, here's how to monitor them:

### 🧾 Log Files
Each model's logs (stdout/stderr) are saved under:

results/logs/<MODEL_NAME>_<JOBID>.out


### 🔍 Live Monitoring Tips

| Task                   | Command |
|------------------------|---------|
| 📖 View live log output | `tail -f results/logs/*.out` |
| 📋 View queued jobs     | `squeue -u $USER` |

---

Make sure your environment is activated and configured before launching or tailing jobs.

## 🧪 Testing a Model

To test a model:


1. Baseline Model

```bash
MODEL_TYPE=gray2real_baseline \
EXPERIMENT_NAME=gray2real_baseline \
EPOCH=60 \
GPU_ID=0 \
NUM_TEST=50 \
./test_model.sh
```
2. Standard gray2real (L1 + cGAN)

```bash

MODEL_TYPE=gray2real \
EXPERIMENT_NAME=gray2real_final \
EPOCH=60 \
GPU_ID=0 \
NUM_TEST=50 \
./test_model.sh
```
3. gray2real with VGG Loss

```bash

MODEL_TYPE=gray2real \
EXPERIMENT_NAME=gray2real_final_vgg \
EPOCH=30 \
GPU_ID=0 \
NUM_TEST=50 \
./test_model.sh
```
⚡ Note: This model was likely trained for 30 + 30 epochs, so adjust EPOCH if needed.

4. gray2real with FaceNet (Identity) Loss
```bash

MODEL_TYPE=gray2real \
EXPERIMENT_NAME=gray2real_final_id \
EPOCH=30 \
GPU_ID=0 \
NUM_TEST=50 \
./test_model.sh
```
5. gray2real with VGG + FaceNet (Identity) Losses

```bash

MODEL_TYPE=gray2real \
EXPERIMENT_NAME=gray2real_final_both \
EPOCH=60 \
GPU_ID=0 \
NUM_TEST=50 \
./test_model.sh
```

## Customizable parameters for testing

| Environment Variable | Description | Default (if any) |
|:---------------------|:-------------|:----------------|
| `MODEL_TYPE`          | Model type to load | Required |
| `EXPERIMENT_NAME`     | Folder name to load checkpoints | Required |
| `EPOCH`               | Which epoch to test (`latest`, `60`, etc.) | `latest` |
| `GPU_ID`              | Which GPU to use | `0` |
| `NUM_TEST`            | Number of images to test | All |
| `DATAROOT`            | Dataset path | Default set inside `test_model.sh` |
| `RESULTS_DIR`         | Where to save outputs | `./results` |
| `PHASE`               | Testing phase | `test` |



### 🧠 Example for Advanced Testing

If you wanted to:

- Use epoch `latest`
- Test 100 images
- Save results somewhere else

You would run:

```bash
MODEL_TYPE=gray2real \
EXPERIMENT_NAME=gray2real_final_vgg \
EPOCH=latest \
GPU_ID=0 \
NUM_TEST=100 \
RESULTS_DIR=./results_vgg \
PHASE=test \
./test_model.sh
```