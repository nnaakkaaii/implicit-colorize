from torch import Tensor, nn

from .resnet_block import ResNetBlock


class ResNetEncoder(nn.Module):
    """ResNetEncoder encodes (1, 96, 96) image into 128-dim vector"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # (16, 48, 48)
            nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.GELU(),
            ResNetBlock(16),
            nn.GELU(),
            # (32, 24, 24)
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            ResNetBlock(32),
            nn.GELU(),
            # (64, 12, 12)
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            ResNetBlock(64),
            nn.GELU(),
            # (128, 6, 6)
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            ResNetBlock(128),
            nn.AvgPool2d(kernel_size=(6, 6)),
            nn.Flatten(),
            )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


if __name__ == "__main__":
    # python3 -m imcolorize.networks.resnet_encoder
    from torch import randn

    net = ResNetEncoder()
    y = net(randn(16, 1, 96, 96))
    print(y.shape)
