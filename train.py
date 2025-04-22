import time
import torch
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
import os
import wandb
import torchvision.utils as vutils

# 🌐 Enable proxy for Hyperion
os.environ["https_proxy"] = "http://hpc-proxy00.city.ac.uk:3128"


# 🧵 Combine sketch, fake, and photo into one horizontal image
def make_side_by_side(sketch, fake, photo, normalize=False):
    """
    🖼️ Create a clean 3-channel comparison image. Works with normalized tensors in [-1, 1].
    """

    if sketch.ndim == 3:
        sketch = sketch.unsqueeze(0)
    if fake.ndim == 3:
        fake = fake.unsqueeze(0)
    if photo.ndim == 3:
        photo = photo.unsqueeze(0)

    if sketch.shape[1] == 1:
        sketch = sketch.repeat(1, 3, 1, 1)  # expand grayscale

    # 🌀 Rescale from [-1, 1] → [0, 1]
    sketch = (sketch + 1) / 2
    fake = (fake + 1) / 2
    photo = (photo + 1) / 2

    # 📏 Resize to match sketch dimensions
    target_size = sketch.shape[-2:]
    fake = torch.nn.functional.interpolate(
        fake, size=target_size, mode="bilinear", align_corners=False
    )
    photo = torch.nn.functional.interpolate(
        photo, size=target_size, mode="bilinear", align_corners=False
    )

    return vutils.make_grid(
        torch.cat([sketch, fake, photo], dim=0),
        nrow=3,
        normalize=False,
        pad_value=255,
    )


def train(opt):
    # 🏷️ Add prefix for logging jobs – use experiment name
    log_prefix = f"[{opt.name}]"

    # 🔵 Set device for GPU/CPU with respect to opt.gpu_ids
    device = torch.device(
        "cuda" if torch.cuda.is_available() and opt.gpu_ids[0] >= 0 else "cpu"
    )

    if device.type == "cuda":
        print(f"{log_prefix} 🚀 Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        print(f"{log_prefix} 🐢 Using CPU")

    # 🔖 Initialize WandB logging with group/tag setup
    wandb.init(
        project=opt.name if hasattr(opt, "name") else "gray2real",
        name=f"{opt.name}_run",
        config=vars(opt),
        group=opt.model,  # Group runs by model name (e.g., gray2real)
        tags=[
            opt.dataset_mode,
            opt.model,
            "gray2real",
        ],  # 🏷️ Add custom tags
        mode="offline" if getattr(opt, "offline", False) else "online",
    )

    # Log image tensors to WandB
    def log_images_to_wandb(visuals, step, label=""):
        """
        🖼️ Logs key visuals + comparison + grouped grid to WandB
        """
        images = []

        # 🔁 INDIVIDUAL LOGGING (each visual key as a separate image)
        for name, img_tensor in visuals.items():
            if isinstance(img_tensor, torch.Tensor):
                grid = vutils.make_grid(img_tensor, normalize=True, scale_each=True)
                wandb.log({f"{name}": wandb.Image(grid)}, step=step)
                images.append(wandb.Image(grid, caption=f"{label}_{name}"))

        # 🧵 SIDE-BY-SIDE COMPARISON (first sketch, fake_B, photo only)
        if all(k in visuals for k in ["real_A", "fake_B", "real_B"]):
            sketch = visuals["real_A"][0].detach().cpu()
            fake = visuals["fake_B"][0].detach().cpu()
            photo = visuals["real_B"][0].detach().cpu()

            # 🧵 Save side-by-side grid for local and wandb
            comparison_grid = make_side_by_side(sketch, fake, photo)
            # vutils.save_image(comparison_grid, f"debug_comparison_epoch{label}.png")

            # ✅ Pass tensor directly — don't convert to PIL!
            wandb.log({"comparison": wandb.Image(comparison_grid)}, step=step)

        # 📊 GROUPED PANEL (logs multiple images under one key)
        if images:
            wandb.log({f"visuals/{label}": images}, step=step)

    # 📦 Load dataset
    dataset = create_dataset(opt)
    dataset_size = len(dataset)
    print(f"{log_prefix} 🖼️ The number of training images = {dataset_size}")

    # 🧱 Create model
    model = create_model(opt)
    model.setup(opt)

    # 🚀 Move model to device
    if hasattr(model, "netG"):
        model.netG.to(device)
    if hasattr(model, "netD"):
        model.netD.to(device)

    visualizer = Visualizer(opt)
    total_iters = 0
    max_epochs = opt.niter + opt.niter_decay + 1
    print(
        f"{log_prefix} 📆 Training will run for {max_epochs - 1} epochs total ({opt.niter} + {opt.niter_decay})"
    )

    # 🔁 Epoch loop
    for epoch in range(opt.epoch_count, max_epochs):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            visualizer.reset()
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size

            # Move input tensors to device
            if isinstance(data, dict):
                for k, v in data.items():
                    if isinstance(v, torch.Tensor):
                        data[k] = v.to(device)

            # 🧠 Run model training step
            model.set_input(data)
            model.optimize_parameters()

            # 🖼️ Log images + losses every N steps
            if total_iters % opt.print_freq == 0:
                visuals = model.get_current_visuals()
                log_images_to_wandb(
                    visuals, step=total_iters, label=f"step_{total_iters}"
                )

                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(
                    epoch, epoch_iter, total_iters, losses, t_comp, t_data
                )

            # 💾 Save latest model periodically
            if total_iters % opt.save_latest_freq == 0:
                print(
                    f"{log_prefix} 💾 Saving latest model (epoch {epoch}, total_iters {total_iters})"
                )
                save_suffix = "iter_%d" % total_iters if opt.save_by_iter else "latest"
                model.save_networks(save_suffix)

            iter_data_time = time.time()

        # 💾 Save checkpoint at the end of the epoch
        if epoch % opt.save_epoch_freq == 0:
            print(f"{log_prefix} 📦 Saving model at epoch {epoch}, iters {total_iters}")
            model.save_networks(epoch)

        print(
            f"{log_prefix} ✅ End of epoch {epoch} / {opt.niter + opt.niter_decay} ⏱️ Time Taken: {int(time.time() - epoch_start_time)} sec"
        )

        # 💾 Save the latest model checkpoint
        model.save_networks("latest")

        # 🖼️ Log visuals once per N epochs
        if epoch % 5 == 0 or epoch == 1:
            visuals = model.get_current_visuals()
            log_images_to_wandb(visuals, step=total_iters, label=f"epoch_{epoch}")

        # 🔄 Update LR
        model.update_learning_rate()
        lr = (
            model.get_current_learning_rate()
            if hasattr(model, "get_current_learning_rate")
            else opt.lr
        )
        total_iters += 1
        wandb.log({"learning_rate": lr}, step=total_iters)


if __name__ == "__main__":
    opt = TrainOptions().parse()
    try:
        train(opt)
    finally:
        wandb.finish()
