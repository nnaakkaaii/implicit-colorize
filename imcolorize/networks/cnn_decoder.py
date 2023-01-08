from torch import Tensor, nn

from .cnn_block import ConvBlock
from .utils import Reshape


class CNNDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            # (128,)
            nn.Linear(128, 128 * 6 * 6),
            # (128 * 6 * 6,)
            Reshape(128, 6, 6),
            # (128, 6, 6)
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            # (64, 12, 12)
            ConvBlock(64, 64),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            # (32, 24, 24)
            ConvBlock(32, 32),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            # (16, 48, 48)
            ConvBlock(16, 16),
            nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2),
            # (8, 96, 96)
            ConvBlock(8, 8),
            nn.Conv2d(8, 1, kernel_size=1),
            nn.Sigmoid(),
            )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


if __name__ == "__main__":
    # python3 -m imcolorize.networks.cnn_decoder
    from torch import randn

    net = CNNDecoder()
    y = net(randn(16, 128))
    print(y.shape)
