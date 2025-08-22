"""
Test script for image-to-image translation using WandB logging.

Loads a trained model from --checkpoints_dir and logs results to WandB.

Example:
    python test.py --dataroot ./datasets/cuhk --name cuhk_gray2real_final --model gray2real
"""

import os
from options.test_options import TestOptions  # Parse test-specific options
from data import create_dataset  # Load dataset
from models import create_model  # Load model
from util import util  #  Image processing utilities
from PIL import Image  # Convert image arrays to PIL format for WandB
from util.logger import Logger  # WandB logger wrapper
import numpy as np
import torch
from collections import OrderedDict


def test(opt):
    # ------------------- Hard-code test-time parameters -------------------
    opt.num_threads = 0  # Use single thread
    opt.batch_size = 1  # One sample per batch
    opt.serial_batches = (
        True  # No shuffle, comment out if results on random images are needed
    )
    opt.no_flip = True  # No flip, comment out if results on flipped images are needed
    opt.display_id = -1

    # ------------------- Create dataset and model -------------------
    dataset = create_dataset(opt)

    print(f"Dataset created with {len(dataset)} samples.")
    model = create_model(opt)  # Create and return model based on opt.model
    model.setup(opt)

    print("Active model components:", model.model_names)

    # ------------------- Initialize WandB logger -------------------
    logger = Logger(
        experiment_name=f"{opt.name}_test",
        project=opt.name,  # logs to same project as training
        config=vars(opt),
    )
    print("WandB run initialized:", logger.run.name)
    print("View your run at:", logger.run.url)

    # ------------------- Set model to evaluation mode if specified -------------------
    if opt.eval:
        model.eval()

    # ------------------- Run test loop -------------------
    for i, data in enumerate(dataset):
        if opt.num_test > 0 and i >= opt.num_test:
            break

        print(f"Iteration {i}")

        model.set_input(data)  # Set test input
        print("Set input.")

        model.test()  # Forward pass
        print("Ran model.test()")

        model.compute_visuals()

        if hasattr(model, "fake_B_2"):
            fake_b2_tensor = getattr(model, "fake_B_2")
            print("fake_B_2 shape:", fake_b2_tensor.shape)

            fake_b2_np = util.tensor2im(fake_b2_tensor)
            print("fake_B_2 numpy shape:", fake_b2_np.shape)
            print("  R channel mean:", fake_b2_np[:, :, 0].mean())
            print("  G channel mean:", fake_b2_np[:, :, 1].mean())
            print("  B channel mean:", fake_b2_np[:, :, 2].mean())

        visuals = OrderedDict()
        for name in model.visual_names:
            if hasattr(model, name):
                val = getattr(model, name)
                if name == "fake_B_2" and getattr(model, "netG_2", None) is None:

                    continue  # Skip fake_B_2 if netG2 is not defined in this model
                visuals[name] = val

        print(f"Model Visuals keys: {list(visuals.keys())}")

        img_path = model.get_image_paths()
        print(f"Image path: {img_path}")

        if not visuals:
            print(f"No visuals returned at step {i}")
            continue

        # Log images to WandB
        images_to_log = {
            name: Image.fromarray(util.tensor2im(image_tensor))
            for name, image_tensor in visuals.items()
        }

        logger.step = i
        logger.log_images(images_to_log, caption_prefix=f"Image {i}: ")
        logger.increment_step()

        # Save images locally for evaluation notebook
        local_save_dir = os.path.join(
            "results", opt.name, f"{opt.phase}_{opt.epoch or 'latest'}", "images"
        )
        os.makedirs(local_save_dir, exist_ok=True)

        for label, image_pil in images_to_log.items():
            filename = f"{i:04d}_{label}.png"
            save_path = os.path.join(local_save_dir, filename)
            image_pil.save(save_path)
            print(f"Saved: {save_path}")

    # ------------------- Finish WandB run -------------------
    logger.finish()
    print("Test run finished. Check WandB for results.")


# ------------------- Get test options and start test -------------------
if __name__ == "__main__":
    opt = TestOptions().parse()  # Get test options
    test(opt)  # Start test
