import torch
from torch import nn
from torch.nn import ConvTranspose2d, BatchNorm2d, LeakyReLU, Tanh, Conv2d, Sigmoid
from torch.nn.utils import spectral_norm


def weights_init(m):
    if m.__class__.__name__.find('Conv') != -1:
        nn.init.xavier_normal_(m.weight.data)
    elif m.__class__.__name__.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(self, z_feautures: int):
        super(Generator, self).__init__()

        self.z_feautures = z_feautures
        self.generator = nn.Sequential(

            ConvTranspose2d(
                in_channels=z_feautures,
                out_channels=128,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False
            ),
            BatchNorm2d(128),
            LeakyReLU(0.2, True),

            ConvTranspose2d(
                in_channels=128,
                out_channels=64,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False
            ),
            BatchNorm2d(64),
            LeakyReLU(0.2, True),

            ConvTranspose2d(
                in_channels=64,
                out_channels=32,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            BatchNorm2d(32),
            LeakyReLU(0.2, True),

            spectral_norm(ConvTranspose2d(
                in_channels=32,
                out_channels=1,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ), dim=None),
            Tanh()
        )

        self.apply(weights_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.generator(x)


class Discriminator(nn.Module):
    def __init__(self, out_dimension: int):
        super(Discriminator, self).__init__()

        self.out_dimension = out_dimension
        self.discriminator = nn.Sequential(
            spectral_norm(Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            )),
            BatchNorm2d(16),
            LeakyReLU(0.2, True),

            spectral_norm(Conv2d(
                in_channels=16,
                out_channels=64,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            )),
            BatchNorm2d(64),
            LeakyReLU(0.2, True),

            spectral_norm(Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            )),
            BatchNorm2d(128),
            LeakyReLU(0.2, True),

            spectral_norm(Conv2d(
                in_channels=128,
                out_channels=out_dimension,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            )),
            Sigmoid()

        )

        self.apply(weights_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # add normal noise to input
        noise_ = torch.randn(x.size()) * 0.1 if self.training else torch.tensor([0])
        noise_ = torch.autograd.Variable(noise_, requires_grad=False).to(x.get_device())
        return self.discriminator(x + noise_)
