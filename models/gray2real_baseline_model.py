import torch
from .base_model import BaseModel
from . import networks


class Gray2RealBaselineModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(norm="batch", netG="unet_256", dataset_mode="fs2k")
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode="vanilla")
            parser.add_argument(
                "--lambda_L1", type=float, default=100.0, help="weight for L1 loss"
            )
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = ["G_GAN", "G_L1", "D_real", "D_fake"]
        self.visual_names = ["real_A", "fake_B", "real_B"]
        self.model_names = ["G", "D"] if self.isTrain else ["G"]
        # ✅ Ensure G_2 is not part of model_names
        if "G_2" in self.model_names:
            self.model_names.remove("G_2")

        self.netG = networks.define_G(
            opt.input_nc,
            opt.output_nc,
            opt.ngf,
            opt.netG,
            opt.norm,
            not opt.no_dropout,
            opt.init_type,
            opt.init_gain,
            self.gpu_ids,
        )

        if self.isTrain:
            self.netD = networks.define_D(
                opt.input_nc + opt.output_nc,
                opt.ndf,
                opt.netD,
                opt.n_layers_D,
                opt.norm,
                opt.init_type,
                opt.init_gain,
                self.gpu_ids,
            )

            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.optimizer_G = torch.optim.Adam(
                self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999)
            )
            self.optimizer_D = torch.optim.Adam(
                self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999)
            )
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
        # Maintain compatibility with shared testing pipeline
        self.netG2 = None
        # Prevent accidental loading or logging of G_2
        if "G_2" in self.model_names:
            self.model_names.remove("G_2")
        if hasattr(self, "netG2"):
            del self.netG2

    def set_input(self, input):
        assert self.opt.direction in ["gray2real", "face2sketch"]
        gray2real = self.opt.direction == "gray2real"
        self.real_A = input["gray" if gray2real else "photo"].to(self.device)
        self.real_B = input["photo" if gray2real else "gray"].to(self.device)
        self.gray_path = input["gray_path"]
        self.photo_path = input["photo_path"]
        self.image_paths = (self.gray_path, self.photo_path)

    def forward(self):
        self.fake_B = self.netG(self.real_A)

    def backward_D(self):
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    # 🧪 Inference methods for testing
    def test(self):
        """Run forward pass and store visuals."""
        with torch.no_grad():
            self.forward()
            self.compute_visuals()

    def compute_visuals(self):
        """Store images for visualization."""
        self.visuals = {
            "real_A": self.real_A,
            "fake_B": self.fake_B,
            "real_B": self.real_B,
        }
