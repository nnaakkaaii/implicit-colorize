from torch import Tensor, cat, nn

from .cnn_block import ConvBlock
from .utils import Reshape


class UNetBlock(nn.Module):
    def __init__(self,
                 inner: nn.Module,
                 ) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, x: Tensor) -> Tensor:
        return cat([self.inner(x), x], dim=1)


class UNetGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # (1, 96, 96)
            ConvBlock(1, 8),
            # (8, 96, 96)
            UNetBlock(nn.Sequential(
                nn.MaxPool2d(2),
                # (8, 48, 48)
                ConvBlock(8, 16),
                # (16, 48, 48)
                UNetBlock(nn.Sequential(
                    nn.MaxPool2d(2),
                    # (16, 24, 24)
                    ConvBlock(16, 32),
                    # (32, 24, 24)
                    UNetBlock(nn.Sequential(
                        nn.MaxPool2d(2),
                        # (32, 12, 12)
                        ConvBlock(32, 64),
                        # (64, 12, 12)
                        UNetBlock(nn.Sequential(
                            nn.MaxPool2d(2),
                            # (64, 6, 6)
                            ConvBlock(64, 128),
                            # (128, 6, 6)
                            nn.Flatten(),
                            # (128 * 6 * 6,)
                            nn.Linear(128 * 6 * 6, 128),
                            # (128,)
                            nn.Tanh(),
                            nn.Linear(128, 128 * 6 * 6),
                            # (128 * 6 * 6,)
                            Reshape(128, 6, 6),
                            # (128, 6, 6)
                            nn.ConvTranspose2d(128, 64,
                                               kernel_size=2, stride=2),
                            # (64, 12, 12)
                            )),
                        # (128, 12, 12)
                        ConvBlock(128, 64),
                        nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
                        # (32, 24, 24)
                        )),
                    # (64, 24, 24)
                    ConvBlock(64, 32),
                    nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
                    # (16, 48, 48)
                    )),
                # (32, 48, 48)
                ConvBlock(32, 16),
                nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2),
                # (8, 96, 96)
                )),
            # (16, 96, 96)
            ConvBlock(16, 8),
            nn.Conv2d(8, 3, kernel_size=1),
            nn.Sigmoid(),
            )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


if __name__ == "__main__":
    # python3 -m imcolorize.networks.unet_generator
    from torch import randn

    net = UNetGenerator()
    y = net(randn(16, 1, 96, 96))
    print(y.shape)
