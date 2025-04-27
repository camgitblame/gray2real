# gray2real Pipeline

This repo contains a training and testing setup for all models from the gray2real projects.


---

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

```


---

## ⚙️ Customizable Parameters for Training

When using `run_model.sh` or `run_all.sh`, you can override the default training settings by passing environment variables. Here's a full list of customizable options:

| Variable            | Description                                                       | Example                         |
|---------------------|-------------------------------------------------------------------|---------------------------------|
| `MODEL_TYPE`        | Model variant to train (must match implemented class name)       | `gray2real`, `gray2real`        |
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

### 🧠 Example Usage

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
Copy
Edit
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
