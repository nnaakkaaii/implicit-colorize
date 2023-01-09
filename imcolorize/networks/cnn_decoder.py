from torch import Tensor, nn

from .cnn_block import ConvBlock
from .utils import Reshape


class CNNDecoder(nn.Module):
    def __init__(self, dim: int = 256) -> None:
        super().__init__()
        assert dim % 16 == 0
        self.net = nn.Sequential(
            # (256,)
            nn.Linear(dim, dim * 6 * 6),
            # (dim * 6 * 6,)
            Reshape(dim, 6, 6),
            # (dim, 6, 6)
            nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2),
            # (dim // 2, 12, 12)
            ConvBlock(dim // 2, dim // 2),
            nn.ConvTranspose2d(dim // 2, dim // 4, kernel_size=2, stride=2),
            # (dim // 4, 24, 24)
            ConvBlock(dim // 4, dim // 4),
            nn.ConvTranspose2d(dim // 4, dim // 8, kernel_size=2, stride=2),
            # (dim // 8, 48, 48)
            ConvBlock(dim // 8, dim // 8),
            nn.ConvTranspose2d(dim // 8, dim // 16, kernel_size=2, stride=2),
            # (dim // 16, 96, 96)
            ConvBlock(dim // 16, dim // 16),
            nn.Conv2d(dim // 16, 1, kernel_size=1),
            nn.Tanh(),
            )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


if __name__ == "__main__":
    # python3 -m imcolorize.networks.cnn_decoder
    from torch import randn

    net = CNNDecoder()
    y = net(randn(16, 256))
    print(y.shape)
