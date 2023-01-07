from torch import Tensor, nn


class ResNetBlock(nn.Module):
    def __init__(self,
                 ch: int,
                 ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ch),
            nn.GELU(),
            nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ch),
            )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.net(x)
