"""This module contains simple helper functions"""

from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
from . import html


def tensor2im(input_image, imtype=np.uint8):
    """Converts a Tensor array into a numpy image array suitable for PIL/WandB."""

    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):
            image_tensor = input_image.detach().cpu()
        else:
            return input_image

        # [B, C, H, W] → remove batch dim if present
        if image_tensor.ndim == 4:
            image_tensor = image_tensor[0]

        image_numpy = image_tensor.float().numpy()

        # Convert grayscale [1, H, W] → RGB [3, H, W]
        if image_numpy.shape[0] == 1:
            image_numpy = np.tile(image_numpy, (3, 1, 1))

        # [C, H, W] → [H, W, C], then rescale [-1, 1] → [0, 255]
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0

        # # 🔍 Print debug info for WandB troubleshooting
        # print(
        #     f"[DEBUG tensor2im] shape={image_numpy.shape}, dtype={image_numpy.dtype}, min={image_numpy.min()}, max={image_numpy.max()}"
        # )
    else:
        image_numpy = input_image

    return image_numpy.astype(imtype)


def diagnose_network(net, name="network"):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print("shape,", x.shape)
    if val:
        x = x.flatten()
        print(
            "mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f"
            % (np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x))
        )


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)


def save_images(webpage, visuals, image_path, aspect_ratio=1.0, width=256):
    """Save images to the disk and optionally update HTML webpage.

    Parameters:
        webpage (the HTML class) -- the HTML webpage class
        visuals (OrderedDict)    -- an ordered dictionary of images to save
        image_path (str)         -- the path to the original image
        aspect_ratio (float)     -- the aspect ratio of saved images
        width (int)              -- the width of saved images
    """
    image_dir = webpage.get_image_dir()
    short_path = os.path.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    webpage.add_header(name)
    ims, txts, links = [], [], []

    for label, image in visuals.items():
        image_numpy = tensor2im(image)
        img_name = f"{name}_{label}.png"
        save_path = os.path.join(image_dir, img_name)
        save_image(image_numpy, save_path)

        ims.append(img_name)
        txts.append(label)
        links.append(img_name)

    webpage.add_images(ims, txts, links, width=width)
