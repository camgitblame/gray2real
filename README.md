# 🧪 gray2real Model Training Pipeline

This repo contains a unified training setup for all models from the gray2real project, including:
- `baseline`
- `gray2real`
- `cgan_initial`, `cgan_final`, `cgan_both`
- `attention`

---

## 🚀 Usage

### ✅ Train a model via SLURM

1. Baseline
If you want to configure the baseline model with:

✅ 200 epochs + 100 decay

✅ Save every epoch

✅ Print every 50 steps

✅ No save-by-iter

✅ Using GPU 0

Run the following command:

```bash
MODEL_TYPE=gray2real_baseline \
EXPERIMENT_NAME=gray2real_baseline_final \
NITER=200 \
NITER_DECAY=100 \
SAVE_EPOCH_FREQ=1 \
PRINT_FREQ=50 \
SAVE_BY_ITER=false \
GPU_ID=0 \
./run_model.sh
```

2. Gray2Real

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
3. cGAN-Final

```bash
MODEL_TYPE=cgan_final \
EXPERIMENT_NAME=cgan_final \
DIRECTION=gray2real \
GPU_ID=0 \
./run_model.sh
```

4. cGAN-Both

```bash
MODEL_TYPE=cgan_both \
EXPERIMENT_NAME=cgan_both \
DIRECTION=gray2real \
GPU_ID=0 \
./run_model.sh

```

5. Self-Attention

```bash
MODEL_TYPE=attention \
EXPERIMENT_NAME=attn_test \
DIRECTION=gray2real \
GPU_ID=0 \
./run_model.sh


```


---

## 🧪 Supported Models & Directions

| MODEL_TYPE     | DIRECTION     | Notes                            |
|----------------|---------------|----------------------------------|
| baseline       | gray2real     | Single U-Net, no cGAN            |
| gray2real      | gray2real     | Grayscale-to-photo               |
| cgan_initial   | gray2real     | G1 = GAN, G2 = L1                |
| cgan_final     | gray2real     | G1 = L1, G2 = GAN                |
| cgan_both      | gray2real     | Both G1 and G2 = GAN             |
| attention      | gray2real     | Self-attention + spectral norm   |

---
## ⚙️ Customizable Parameters for Training

When using `run_model.sh` or `run_all.sh`, you can override the default training settings by passing environment variables. Here's a full list of customizable options:

| Variable            | Description                                                       | Example                         |
|---------------------|-------------------------------------------------------------------|---------------------------------|
| `MODEL_TYPE`        | Model variant to train (must match implemented class name)       | `gray2real`, `cgan_both`        |
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
| ✅ View completed jobs  | `sacct -u $USER --format=JobID,JobName,State,Elapsed` |
| 📊 View WandB logs      | [wandb.ai](https://wandb.ai) (search by project or run name) |

---

Make sure your environment is activated and configured before launching or tailing jobs.

## 🧪 Testing a Trained Model

To test a specific checkpoint:
```bash
python test.py \
  --dataroot ./datasets/cuhk \
  --name gray2real_final \
  --model gray2real \
  --epoch 300
```

To test the latest checkpoint:
```bash
python test.py \
  --dataroot ./datasets/cuhk \
  --name gray2real_final \
  --model gray2real \
  --epoch latest
```