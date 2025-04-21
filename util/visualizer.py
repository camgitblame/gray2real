import os
from PIL import Image
from . import util
from util.logger import Logger


class Visualizer:
    def __init__(self, opt):
        self.opt = opt
        self.name = opt.name
        self.logger = Logger(
            experiment_name=opt.name, project=opt.name, config=vars(opt)
        )
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, "loss_log.txt")
        os.makedirs(os.path.dirname(self.log_name), exist_ok=True)
        self.saved = False  # Flag to track if results were saved

    def reset(self):
        """Reset the saved flag at the start of each iteration."""
        self.saved = False

    def display_current_results(self, visuals, epoch, save_result):
        """Log current visual outputs as images to WandB."""
        images_to_log = {}
        for label, image_tensor in visuals.items():
            if image_tensor is not None:
                image_numpy = util.tensor2im(image_tensor)  # Convert to numpy array
                pil_image = Image.fromarray(image_numpy)  # Convert to PIL Image
                images_to_log[label] = pil_image
        self.logger.log_images(images_to_log, caption_prefix=f"Epoch {epoch} ")

    def display_side_by_side(self, sketch, fake, photo, step, label="comparison"):
        """🧵 Log a single comparison image (sketch | fake | photo) to WandB."""
        self.logger.log_comparison(sketch, fake, photo, label)

    def print_current_losses(self, epoch, iters, total_iters, losses, t_comp, t_data):
        """Print training loss values to console and file."""
        message = (
            f"(epoch: {epoch}, iters: {iters}, time: {t_comp:.3f}, data: {t_data:.3f}) "
        )
        message += " ".join([f"{k}: {v:.3f}" for k, v in losses.items()])
        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write(message + "\n")

        # ✅ Onl log tracked losses to keep WandB clean
        tracked = ["G_GAN", "G_L1", "G_2_L1", "D_real", "D_fake"]
        log_dict = {f"loss/{k}": v for k, v in losses.items() if k in tracked}
        self.logger.log(log_dict, step=total_iters)
        self.logger.increment_step()
