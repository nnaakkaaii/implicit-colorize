from typing import List, Optional, Tuple

from torch import Tensor, cat, nn, tensor

from .imnet_block import IMNetBlock


class IMNetDecoder(nn.Module):
    def __init__(self,
                 img_size: int = 96,
                 dim: int = 256,
                 ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            IMNetBlock(
                dim * 4,
                dim * 2,
                IMNetBlock(
                    dim * 8,
                    dim * 4,
                    IMNetBlock(
                        dim * 16,
                        dim * 8,
                        IMNetBlock(
                            dim + 2,
                            dim * 16,
                            ),
                        ),
                    ),
                ),
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, 3),
            # nn.Tanh(),
            )
        self.img_size = img_size

    def forward(self,
                x: Tensor,
                c: Optional[List[Tuple[int, int]]] = None,
                ) -> Tensor:
        bs = x.size(0)

        if c is None:
            c = [(i / self.img_size, j / self.img_size)
                 for i in range(self.img_size)
                 for j in range(self.img_size)
                 ]
            x = x.repeat_interleave(
                self.img_size * self.img_size,
                dim=0,
                )

        t_c = tensor(c).repeat(bs, 1)
        x = cat([x, t_c.to(x)], dim=1)
        y = self.net(x)

        if y.size(0) == bs:
            return y

        return y.view(bs, -1, self.img_size, self.img_size)


if __name__ == "__main__":
    # python3 -m imcolorize.networks.imnet_decoder
    from torch import randn

    net = IMNetDecoder()
    pred = net(randn(16, 256), [(1, 2)])
    print(pred.shape)

    pred = net(randn(16, 256))
    print(pred.shape)
