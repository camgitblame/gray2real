import torch
from .base_model import BaseModel
from . import networks


class Gray2RealModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(norm="batch", netG="unet_256", dataset_mode="fs2k")
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode="vanilla")
            parser.add_argument(
                "--lambda_L1", type=float, default=100.0, help="weight for L1 loss"
            )
            parser.add_argument(
                "--discriminator_norm",
                type=str,
                default="batch",
                help="norm for discriminator. options: [instance | batch | spectral | none]",
            )

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = [
            "G_GAN",
            "G_L1",
            "G_2_L1",
            "G_VGG",  # only if loss_mode includes vgg
            "G_ID",  # only if loss_mode includes id
            "D_real",
            "D_fake",
        ]
        self.visual_names = ["real_A", "fake_B", "fake_B_2", "real_B"]
        if self.isTrain:
            self.model_names = ["G", "G_2", "D"]
        else:
            self.model_names = ["G", "G_2"]

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

        self.netG_2 = networks.define_G(
            opt.output_nc,
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
                opt.discriminator_norm,
                opt.init_type,
                opt.init_gain,
                self.gpu_ids,
            )

        if self.isTrain:
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # Load pretrained VGG feature extractor
            if opt.loss_mode in ["vgg", "both"]:
                from torchvision.models import vgg19

                self.vgg = vgg19(pretrained=True).features[:16].eval().to(self.device)
                for param in self.vgg.parameters():
                    param.requires_grad = False

            # Load FaceNet model (custom wrapper or preloaded model)
            if opt.loss_mode in ["id", "both"]:
                from models.facenet import FaceNet  # customized FaceNet loader

                self.facenet = FaceNet().eval().to(self.device)
                for param in self.facenet.parameters():
                    param.requires_grad = False

            self.optimizer_G = torch.optim.Adam(
                self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999)
            )
            self.optimizer_G_2 = torch.optim.Adam(
                self.netG_2.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999)
            )
            self.optimizer_D = torch.optim.Adam(
                self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999)
            )
            self.optimizers += [self.optimizer_G, self.optimizer_G_2, self.optimizer_D]

    def set_input(self, input):
        assert self.opt.direction in ["gray2real", "real2gray"]
        gray2real = self.opt.direction == "gray2real"
        self.real_A = input["gray" if gray2real else "photo"].to(self.device)
        self.real_B = input["photo" if gray2real else "gray"].to(self.device)
        self.gray_path = input["gray_path"]
        self.photo_path = input["photo_path"]
        self.image_paths = (self.gray_path, self.photo_path)

    def forward(self):
        self.fake_B = self.netG(self.real_A)
        self.fake_B_2 = self.netG_2(self.fake_B)

    def compute_visuals(self):
        # print("🎨 [gray2real_model] compute_visuals() called")
        self.visuals = {
            "real_A": self.real_A,
            "fake_B": self.fake_B,
            "fake_B_2": self.fake_B_2,
            "real_B": self.real_B,
        }

    def backward_D(self):
        fake_AB = torch.cat((self.real_A, self.fake_B_2), 1)
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        fake_AB = torch.cat((self.real_A, self.fake_B_2), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        self.loss_G_2_L1 = (
            self.criterionL1(self.fake_B_2, self.real_B) * self.opt.lambda_L1
        )
        # VGG Loss (feature-level perceptual loss)
        if self.opt.loss_mode in ["vgg", "both"]:
            vgg_feat_fake = self.vgg(self.fake_B_2)
            vgg_feat_real = self.vgg(self.real_B)
            self.loss_G_VGG = (
                self.criterionL1(vgg_feat_fake, vgg_feat_real) * self.opt.lambda_vgg
            )
        else:
            self.loss_G_VGG = torch.tensor(0.0, device=self.device)

        # Identity Loss (FaceNet feature similarity)
        if self.opt.loss_mode in ["id", "both"]:
            id_fake = self.facenet(self.fake_B_2)
            id_real = self.facenet(self.real_B)
            self.loss_G_ID = (
                torch.nn.functional.l1_loss(id_fake, id_real) * self.opt.lambda_id
            )
        else:
            self.loss_G_ID = torch.tensor(0.0, device=self.device)

        self.loss_G = (
            self.loss_G_GAN
            + self.loss_G_L1
            + self.loss_G_2_L1
            + self.loss_G_VGG
            + self.loss_G_ID
        )
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.optimizer_G_2.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        self.optimizer_G_2.step()

    def test(self):
        """Run forward and compute visuals (for inference)."""
        # print("[🔥 test()] Running inference + visuals")
        with torch.no_grad():
            self.forward()
            self.compute_visuals()
            # print("✅ gray2real_model.test() called")
