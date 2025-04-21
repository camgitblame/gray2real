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

```bash
MODEL_TYPE=gray2real_baseline \
EXPERIMENT_NAME=gray2real_baseline \
DIRECTION=gray2real \
GPU_ID=0 \
./run_model.sh
```

2. Gray2Real

```bash
MODEL_TYPE=gray2real \
EXPERIMENT_NAME=gray2real_final \
DIRECTION=gray2real \
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

## 🛠️ Advanced

### ✅ Train all models at once

```bash
bash ./run_all.sh
```


