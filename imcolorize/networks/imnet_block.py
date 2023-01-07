from typing import Optional

from torch import Tensor, cat, nn


class IMNetBlock(nn.Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 inner: Optional[nn.Module] = None,
                 org_dim: int = 130,
                 ) -> None:
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(in_dim, out_dim - org_dim),
            nn.GELU(),
            )
        self.inner = inner

    def forward(self, x: Tensor) -> Tensor:
        if self.inner is None:
            return cat([self.linear(x), x], dim=1)
        return cat([self.linear(self.inner(x)), x], dim=1)


if __name__ == "__main__":
    from torch import randn

    net = IMNetBlock(
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
        )
    y = net(randn(16, 130))
    print(y.shape)
