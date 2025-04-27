import os
import re
from PIL import Image
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_params, get_transform


class CuhkDataset(BaseDataset):
    """📚 Custom CUHK Dataset for sketch-to-photo translation"""

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """
        ⚙️ Add dataset-specific CLI options (used in --dataset_mode cuhk)

        Args:
            parser: argparse parser from BaseOptions
            is_train: bool flag for training vs test mode

        Returns:
            parser with additional options
        """
        parser.add_argument(
            "--max_rotation",
            type=float,
            default=10.0,
            help="🌀 Max rotation for augmentation",
        )
        parser.add_argument(
            "--min_crop_scale",
            type=float,
            default=0.8,
            help="🔍 Min scale for random resized crop",
        )
        parser.add_argument(
            "--max_crop_scale",
            type=float,
            default=1.0,
            help="🔍 Max scale for random resized crop",
        )
        return parser

    @staticmethod
    def make_pairs_dataset(root, max_dataset_size=float("inf")):
        """
        🔗 Match sketch-photo pairs using filenames like 'f-005-01-sz1.jpg' and 'f-005-01.jpg'
        """
        pairs = []
        files = sorted(os.listdir(root))

        # 🧠 Find all _sz1.jpg files and match with corresponding .jpg files
        for fname in files:
            if fname.endswith("-sz1.jpg"):
                sketch_path = os.path.join(root, fname)
                photo_fname = fname.replace(
                    "-sz1", ""
                )  # Remove -sz1 to get the real photo filename
                photo_path = os.path.join(root, photo_fname)

                if os.path.exists(photo_path):
                    pairs.append((sketch_path, photo_path))

        return pairs[: min(max_dataset_size, len(pairs))]

    def __init__(self, opt):
        """
        🚀 Initialize the CUHK dataset

        Args:
            opt: argparse-style options from BaseOptions
        """
        # Save options and dataset root
        BaseDataset.__init__(self, opt)
        self.dir_images = os.path.join(
            opt.dataroot, opt.phase
        )  # 📂 e.g. ./datasets/cuhk/train
        self.image_pair_paths = sorted(
            self.make_pairs_dataset(self.dir_images, opt.max_dataset_size)
        )

        # 🧠 Set input/output channels from CLI args
        self.input_nc = opt.input_nc  # sketch channel (1=grayscale, 3=RGB)
        self.output_nc = opt.output_nc  # photo channel (typically RGB)

    def __getitem__(self, index):
        """
        📦 Return one paired data point

        Args:
            index: index of the data pair

        Returns:
            dict with sketch, photo tensors + file paths
        """
        sketch_path, photo_path = self.image_pair_paths[index]

        # 🖼️ Load images — grayscale or RGB depending on input/output channels
        sketch = Image.open(sketch_path).convert("L" if self.input_nc == 1 else "RGB")
        photo = Image.open(photo_path).convert("L" if self.output_nc == 1 else "RGB")

        # 🔁 Use shared transform parameters for both images
        transform_params = get_params(self.opt, sketch.size)
        sketch_transform = get_transform(
            self.opt, transform_params, grayscale=(self.input_nc == 1)
        )
        photo_transform = get_transform(
            self.opt, transform_params, grayscale=(self.output_nc == 1)
        )

        # 🧪 Apply transforms
        sketch = sketch_transform(sketch)
        photo = photo_transform(photo)

        return {
            "gray": sketch,
            "photo": photo,
            "gray_path": sketch_path,
            "photo_path": photo_path,
        }

    def __len__(self):
        """🔢 Return number of valid sketch-photo pairs"""
        return len(self.image_pair_paths)
