from torch import Tensor, nn

from .cnn_block import ConvBlock


class CNNEncoder(nn.Module):
    """CNNEncoder encodes (1, 96, 96) image into 256-dim vector"""
    def __init__(self, dim: int = 256):
        super().__init__()
        assert dim % 16 == 0
        self.net = nn.Sequential(
            # (1, 96, 96)
            ConvBlock(1, dim // 16),
            # (16, 96, 96)
            nn.MaxPool2d(2),
            # (16, 48, 48)
            ConvBlock(dim // 16, dim // 8),
            # (32, 48, 48)
            nn.MaxPool2d(2),
            # (32, 24, 24)
            ConvBlock(dim // 8, dim // 4),
            # (64, 24, 24)
            nn.MaxPool2d(2),
            # (64, 12, 12)
            ConvBlock(dim // 4, dim // 2),
            # (128, 12, 12)
            nn.MaxPool2d(2),
            # (128, 6, 6)
            ConvBlock(dim // 2, dim),
            # (256, 6, 6)
            nn.Flatten(),
            # (256 * 6 * 6,)
            nn.Linear(dim * 6 * 6, dim),
            # (256,)
            nn.Tanh(),
            )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


if __name__ == "__main__":
    # python3 -m imcolorize.networks.cnn_encoder
    from torch import randn

    net = CNNEncoder()
    y = net(randn(16, 1, 96, 96))
    print(y.shape)
