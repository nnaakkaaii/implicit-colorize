from torch import Tensor, nn


class ConvBlock(nn.Module):
    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch,
                      out_ch,
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      ),
            nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class IdConvBlock(nn.Module):
    def __init__(self,
                 ch: int,
                 ) -> None:
        super().__init__()
        self.net = nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class DeconvBlock(nn.Module):
    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_ch,
                               out_ch,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               ),
            nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
