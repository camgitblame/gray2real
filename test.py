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
from util.util import save_images
from util import html  # 👈 for HTML logging


def test(opt):
    # ------------------- ⚙️ Hard-code test-time parameters -------------------
    opt.num_threads = 0  # Use single thread
    opt.batch_size = 1  # One sample per batch
    opt.serial_batches = (
        True  # No shuffle, comment out if results on random images are needed
    )
    opt.no_flip = True  # No flip, comment out if results on flipped images are needed
    opt.display_id = -1

    # ------------------- 📚 Create dataset and model -------------------
    dataset = create_dataset(opt)

    print(f"🧪 Dataset created with {len(dataset)} samples.")
    model = create_model(opt)  # Create and return model based on opt.model
    model.setup(opt)

    web_dir = os.path.join(
        "./results", opt.name, f"{opt.phase}_{opt.epoch or 'latest'}"
    )

    webpage = html.HTML(web_dir, f"Experiment = {opt.name}", refresh=1)

    # ------------------- 🚀 Initialize WandB logger -------------------
    logger = Logger(
        experiment_name=f"{opt.name}_test",
        project=opt.name,  # ✅ logs to same project as training
        config=vars(opt),
    )
    print("🟢 WandB run initialized:", logger.run.name)
    print("🔗 View your run at:", logger.run.url)

    # ------------------- 📥 Set model to evaluation mode if specified -------------------
    if opt.eval:
        model.eval()

    # ------------------- 🔁 Run test loop -------------------
    for i, data in enumerate(dataset):
        if opt.num_test > 0 and i >= opt.num_test:
            break

        print(f"🔄 Iteration {i}")

        model.set_input(data)  # Set test input
        print("✅ Set input.")

        model.test()  # Forward pass
        print("✅ Ran model.test()")

        model.compute_visuals()

        visuals = model.get_current_visuals()
        print(f"🔍 Visuals keys: {list(visuals.keys())}")

        img_path = model.get_image_paths()
        print(f"📂 Image path: {img_path}")

        # 🖼️ Save visual outputs to HTML + PNGs
        save_images(
            webpage,
            visuals,
            img_path,
            aspect_ratio=opt.aspect_ratio,
            width=opt.display_winsize,
        )

        if not visuals:
            print(f"🚫 No visuals returned at step {i}")
            continue

        # 📝 Log images to WandB
        images_to_log = {}
        for label, image_tensor in visuals.items():
            # print(f"📸 Processing {label}...")
            image_numpy = util.tensor2im(image_tensor)
            image_pil = Image.fromarray(image_numpy)
            images_to_log[label] = image_pil
            # print(f"🖼️ {label} ready. Size: {image_pil.size}")

        logger.step = i
        # 🧪 Sanity check on images before WandB log
        # for label, image in images_to_log.items():
        #     print(
        #         f"[CHECK] {label}: type={type(image)}, size={getattr(image, 'size', 'N/A')}"
        #     )

        logger.log_images(images_to_log, caption_prefix=f"Image {i}: ")
        logger.increment_step()
        # print(f"✅ Logged image {i} to WandB\n")

    webpage.save()

    # ------------------- Finish WandB run -------------------
    logger.finish()
    print("🎉 Test run finished. Check WandB for results.")


# ------------------- ▶️ Get test options and start test -------------------
if __name__ == "__main__":
    opt = TestOptions().parse()  # Get test options
    test(opt)  # Start test
