# BaseOptions defines and parses command-line arguments used for training and testing.
import argparse  # For parsing command-line arguments
import os  # For path operations
from util import util  # Utility functions like mkdirs
import torch  # PyTorch for model setup
import models  # Custom models directory
import data  # Custom dataset loader


class BaseOptions:
    """Defines common options and utility methods for training and test phases."""

    def __init__(self):
        """Initialization flag to avoid re-initializing."""
        self.initialized = False

    def initialize(self, parser):
        """Add common options to the parser."""
        # Basic parameters
        parser.add_argument(
            "--dataroot",
            required=True,
            help="Path to datasets with subfolders like trainA, trainB, etc.",
        )
        parser.add_argument(
            "--name",
            type=str,
            default="experiment_name",
            help="Name of the experiment (used for checkpoints and logs).",
        )
        parser.add_argument(
            "--gpu_ids", type=str, default="0", help="GPU IDs to use, -1 for CPU."
        )
        parser.add_argument(
            "--checkpoints_dir",
            type=str,
            default="./checkpoints",
            help="Directory to save models.",
        )

        # Model parameters
        parser.add_argument(
            "--model",
            type=str,
            default="cycle_gan",
            help="Model type to use: [cycle_gan | pix2pix | test | colorization].",
        )
        parser.add_argument(
            "--input_nc",
            type=int,
            default=3,
            help="Number of input channels (3 for RGB, 1 for grayscale).",
        )
        parser.add_argument(
            "--output_nc",
            type=int,
            default=3,
            help="Number of output channels (3 for RGB, 1 for grayscale).",
        )
        parser.add_argument(
            "--ngf",
            type=int,
            default=64,
            help="Number of generator filters in the last conv layer.",
        )
        parser.add_argument(
            "--ndf",
            type=int,
            default=64,
            help="Number of discriminator filters in the first conv layer.",
        )
        parser.add_argument(
            "--netD",
            type=str,
            default="basic",
            help="Discriminator architecture [basic | n_layers | pixel].",
        )
        parser.add_argument(
            "--netG",
            type=str,
            default="resnet_9blocks",
            help="Generator architecture [resnet_9blocks | resnet_6blocks | unet_256 | unet_128].",
        )
        parser.add_argument(
            "--n_layers_D", type=int, default=3, help="Only used if netD==n_layers."
        )
        parser.add_argument(
            "--self_attn_layers_D",
            nargs="+",
            type=int,
            default=[],
            help="Indices of discriminator layers with self-attention.",
        )
        parser.add_argument(
            "--self_attn_layers_G",
            nargs="+",
            type=int,
            default=[],
            help="Indices of generator layers with self-attention.",
        )
        parser.add_argument(
            "--self_attn_layers_G2",
            nargs="+",
            type=int,
            default=[],
            help="Indices of 2nd generator layers with self-attention.",
        )
        parser.add_argument(
            "--norm",
            type=str,
            default="instance",
            help="Normalization method [instance | batch | none].",
        )
        parser.add_argument(
            "--init_type",
            type=str,
            default="normal",
            help="Weight initialization method.",
        )
        parser.add_argument(
            "--init_gain",
            type=float,
            default=0.02,
            help="Gain for weight initialization.",
        )
        parser.add_argument(
            "--no_dropout", action="store_true", help="Disable dropout for generator."
        )

        # Dataset parameters
        parser.add_argument(
            "--dataset_mode",
            type=str,
            default="unaligned",
            help="Dataset loading method [unaligned | aligned | single | colorization].",
        )
        parser.add_argument(
            "--direction",
            type=str,
            default="AtoB",
            help="Conversion direction: AtoB or BtoA.",
        )
        parser.add_argument(
            "--serial_batches",
            action="store_true",
            help="Load images in order rather than randomly.",
        )
        parser.add_argument(
            "--num_threads", default=4, type=int, help="Number of data loading threads."
        )
        parser.add_argument(
            "--batch_size", type=int, default=1, help="Input batch size."
        )
        parser.add_argument(
            "--load_size", type=int, default=286, help="Resize images to this size."
        )
        parser.add_argument(
            "--crop_size", type=int, default=256, help="Crop images to this size."
        )
        parser.add_argument(
            "--max_dataset_size",
            type=int,
            default=float("inf"),
            help="Maximum number of samples to load.",
        )
        parser.add_argument(
            "--preprocess",
            type=str,
            default="resize_and_crop",
            help="Image preprocessing method.",
        )
        parser.add_argument(
            "--no_flip",
            action="store_true",
            help="Disable image flipping for data augmentation.",
        )

        # Additional parameters
        parser.add_argument(
            "--epoch",
            type=str,
            default="latest",
            help="Epoch to load for resuming training.",
        )
        parser.add_argument(
            "--load_iter",
            type=int,
            default="0",
            help="Iteration to load if greater than zero.",
        )
        parser.add_argument(
            "--verbose", action="store_true", help="Enable verbose debugging output."
        )
        parser.add_argument(
            "--suffix", default="", type=str, help="Suffix to add to experiment name."
        )

        self.initialized = True
        return parser

    def gather_options(self):
        """Gathers and finalizes command-line arguments."""
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )
            parser = self.initialize(parser)

        opt, _ = parser.parse_known_args()

        # Update parser with model-specific options
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)
        opt, _ = parser.parse_known_args()

        # Update parser with dataset-specific options
        dataset_name = opt.dataset_mode
        dataset_option_setter = data.get_option_setter(dataset_name)
        parser = dataset_option_setter(parser, self.isTrain)

        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        """Prints and saves parsed options to disk."""
        message = "----------------- Options ---------------\n"
        for k, v in sorted(vars(opt).items()):
            comment = ""
            default = self.parser.get_default(k)
            if v != default:
                comment = f"\t[default: {default}]"
            message += f"{k:>25}: {str(v):<30}{comment}\n"
        message += "----------------- End -------------------"
        print(message)

        # Save to disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, f"{opt.phase}_opt.txt")
        with open(file_name, "wt") as opt_file:
            opt_file.write(message + "\n")

    def parse(self):
        """Parses options and sets GPU devices."""
        opt = self.gather_options()
        opt.isTrain = self.isTrain

        # Apply suffix if provided
        if opt.suffix:
            suffix = f"_{opt.suffix.format(**vars(opt))}" if opt.suffix != "" else ""
            opt.name = opt.name + suffix

        self.print_options(opt)

        # GPU setup
        str_ids = opt.gpu_ids.split(",")
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt
