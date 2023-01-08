from torch import Tensor, nn

from .cnn_block import DeconvBlock


class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = tuple(args)

    def forward(self, x: Tensor) -> Tensor:
        return x.view(-1, *self.shape)


class CNNDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            Reshape(128, 1, 1),
            nn.ConvTranspose2d(128, 128, kernel_size=6),
            nn.ELU(),
            DeconvBlock(128, 64),
            nn.ELU(),
            DeconvBlock(64, 32),
            nn.ELU(),
            DeconvBlock(32, 16),
            nn.ELU(),
            DeconvBlock(16, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


if __name__ == "__main__":
    # python3 -m imcolorize.networks.cnn_decoder
    from torch import randn

    net = CNNDecoder()
    y = net(randn(16, 128))
    print(y.shape)
