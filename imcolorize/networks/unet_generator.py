from torch import Tensor, cat, nn

from .cnn_block import ConvBlock, DeconvBlock, IdConvBlock


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
            # (16, 48, 48)
            ConvBlock(1, 16),
            nn.GELU(),
            UNetBlock(nn.Sequential(
                # (32, 24, 24)
                ConvBlock(16, 32),
                nn.GELU(),
                UNetBlock(nn.Sequential(
                    # (64, 12, 12)
                    ConvBlock(32, 64),
                    nn.GELU(),
                    UNetBlock(nn.Sequential(
                        # (128, 6, 6)
                        ConvBlock(64, 128),
                        nn.GELU(),
                        IdConvBlock(128),
                        nn.ELU(),
                        DeconvBlock(128, 64),
                        )),
                    nn.ELU(),
                    DeconvBlock(128, 32),
                    )),
                nn.ELU(),
                DeconvBlock(64, 16),
                )),
            nn.ELU(),
            DeconvBlock(32, 3),
            )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


if __name__ == '__main__':
    # python3 -m imcolorize.networks.unet_generator
    from torch import randn

    net = UNetGenerator()
    y = net(randn(16, 1, 96, 96))
    print(y.shape)
