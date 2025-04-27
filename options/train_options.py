from .base_options import (
    BaseOptions,
)  # Import the base class that contains shared options


class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        # Initialize with base options first (common to both training and testing)
        parser = BaseOptions.initialize(self, parser)

        # ---------------------- Logging ----------------------

        # Frequency (in steps) to print training losses to the console and log to WandB
        parser.add_argument(
            "--print_freq",
            type=int,
            default=100,
            help="Frequency of logging losses and images to console and WandB.",
        )

        # ---------------------- Checkpoint Saving ----------------------

        # Frequency (in iterations) to save the latest model
        parser.add_argument(
            "--save_latest_freq",
            type=int,
            default=5000,
            help="Frequency of saving the latest model (in iterations).",
        )

        # Frequency (in epochs) to save model checkpoints
        parser.add_argument(
            "--save_epoch_freq",
            type=int,
            default=5,
            help="Frequency of saving model checkpoints (in epochs).",
        )

        # If True, save by iteration number instead of just 'latest'
        parser.add_argument(
            "--save_by_iter",
            action="store_true",
            help="If set, save model checkpoints with iteration number.",
        )

        # If True, continue training from latest saved checkpoint
        parser.add_argument(
            "--continue_train",
            action="store_true",
            help="Continue training: load the latest model checkpoint.",
        )

        # Starting epoch number (for resumed training or custom count)
        parser.add_argument(
            "--epoch_count",
            type=int,
            default=1,
            help="Starting epoch count (used for resumed training).",
        )

        # Phase of operation
        parser.add_argument(
            "--phase",
            type=str,
            default="train",
            help="Current phase of operation (e.g., train, test, val).",
        )

        # ---------------------- Training Hyperparameters ----------------------

        parser.add_argument(
            "--niter",
            type=int,
            default=100,
            help="# of epochs with the initial learning rate.",
        )

        parser.add_argument(
            "--niter_decay",
            type=int,
            default=100,
            help="# of epochs to linearly decay learning rate to zero.",
        )

        parser.add_argument(
            "--beta1",
            type=float,
            default=0.5,
            help="Momentum term (beta1) for Adam optimizer.",
        )

        parser.add_argument(
            "--lr",
            type=float,
            default=0.0002,
            help="Initial learning rate for Adam optimizer.",
        )

        parser.add_argument(
            "--gan_mode",
            type=str,
            default="lsgan",
            help="GAN loss mode. [vanilla | lsgan | wgangp]",
        )

        parser.add_argument(
            "--pool_size",
            type=int,
            default=50,
            help="Buffer size of previously generated images for GAN training.",
        )

        parser.add_argument(
            "--lr_policy",
            type=str,
            default="linear",
            help="Learning rate scheduling policy. [linear | step | plateau | cosine]",
        )

        parser.add_argument(
            "--lr_decay_iters",
            type=int,
            default=50,
            help="Decay learning rate every this many iterations (for 'step' policy).",
        )

        parser.add_argument(
            "--max_epochs",
            type=int,
            default=1000000,
            help="Maximum number of epochs to run. Will stop earlier based on niter + niter_decay.",
        )

        # ---------------------- Loss Mode & Weights ----------------------

        parser.add_argument(
            "--loss_mode",
            type=str,
            default="default",
            choices=["default", "vgg", "id", "both"],
            help="Which extra loss to apply: VGG, identity (FaceNet), or both.",
        )

        parser.add_argument(
            "--lambda_vgg",
            type=float,
            default=10.0,
            help="Weight for VGG perceptual loss.",
        )

        parser.add_argument(
            "--lambda_id",
            type=float,
            default=5.0,
            help="Weight for identity loss (FaceNet embedding distance).",
        )

        # Set training mode flag
        self.isTrain = True

        return parser
