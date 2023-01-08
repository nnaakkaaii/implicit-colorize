from torch import Tensor, nn

from .cnn_block import ConvBlock


class CNNEncoder(nn.Module):
    """CNNEncoder encodes (1, 96, 96) image into 128-dim vector"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # (1, 96, 96)
            ConvBlock(1, 8),
            # (8, 96, 96)
            nn.MaxPool2d(2),
            # (8, 48, 48)
            ConvBlock(8, 16),
            # (16, 48, 48)
            nn.MaxPool2d(2),
            # (16, 24, 24)
            ConvBlock(16, 32),
            # (32, 24, 24)
            nn.MaxPool2d(2),
            # (32, 12, 12)
            ConvBlock(32, 64),
            # (64, 12, 12)
            nn.MaxPool2d(2),
            # (64, 6, 6)
            ConvBlock(64, 128),
            # (128, 6, 6)
            nn.Flatten(),
            # (128 * 6 * 6,)
            nn.Linear(128 * 6 * 6, 128),
            # (128,)
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
