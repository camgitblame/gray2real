import wandb
import torch
import torchvision.utils as vutils


class Logger:
    def __init__(
        self,
        experiment_name,
        project="gray2real",
        config=None,
        group=None,
        tags=None,
        mode="online",
    ):
        # 🚀 Initialize WandB run with all metadata
        self.run = wandb.init(
            project=project,
            name=experiment_name,
            config=config or {},
            group=group,  # Group runs (e.g. per model type)
            tags=tags or [],  # Custom tags (e.g. 'baseline', 'final')
            mode=mode,  # 🌐 Set mode: "online" or "offline"
        )
        self.step = 0  # 🔢 Internal step tracker for consistent logging

        # 🧠 Log device info at init
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device_name = (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
        )
        wandb.config.update({"device": device_name})  # 📋 Store device info in config
        print(f"📝 Logged device to WandB: {device_name}")

    def log(self, metrics: dict, step=None):
        if step is None:
            step = self.step
            self.step += 1
        wandb.log(metrics, step=step)

    # Log all images with optional caption prefix
    def log_images(self, image_dict: dict, caption_prefix=""):
        # 🖼️ Log a dictionary of images with labeled captions
        log_dict = {
            label: wandb.Image(img, caption=f"{caption_prefix}{label}")
            for label, img in image_dict.items()
        }
        wandb.log(log_dict, step=self.step)
        self.step += 1  # 🔁 Increment step to prevent WandB drop warnings 🚨

    # Side-by-side image comparison
    def log_comparison(self, sketch, fake, photo, label="comparison"):
        # 🧩 Combine sketch, generated, and real images in a comparison grid
        grid = vutils.make_grid(
            torch.cat(
                [sketch, fake, photo], dim=0
            ),  # 🪄 Concatenate along vertical axis
            nrow=3,  # 📐 Arrange in a row: [sketch | fake | real]
            normalize=True,
            scale_each=True,
            pad_value=255,
        )
        wandb.log({label: wandb.Image(grid)}, step=self.step)
        self.step += 1  # 🔁 Always increment step for visual logs too 🖼️✅

    def increment_step(self):
        # ➕ Manually bump step counter if needed (e.g. custom loop)
        self.step += 1

    def finish(self):
        # ✅ Finalize and close the WandB run
        self.run.finish()
