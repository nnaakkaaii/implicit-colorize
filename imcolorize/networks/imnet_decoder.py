from typing import Tuple

from torch import Tensor, cat, nn, tensor

from .imnet_block import IMNetBlock


class IMNetDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            IMNetBlock(
                512,
                256,
                IMNetBlock(
                    1024,
                    512,
                    IMNetBlock(
                        2048,
                        1024,
                        IMNetBlock(
                            130,
                            2048,
                            ),
                        ),
                    ),
                ),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

    def forward(self, x: Tensor, c: Tuple[int, int]) -> Tensor:
        t_c = tensor(c).repeat(x.size(0), 1)
        x = cat([x, t_c.to(x)], dim=1)
        return self.net(x)


if __name__ == "__main__":
    # python3 -m imcolorize.networks.imnet_decoder
    from torch import randn

    net = IMNetDecoder()
    y = net(randn(16, 128), (1, 2))
    print(y.shape)
