from typing import Optional

from torch import Tensor, cat, nn


class IMNetBlock(nn.Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 inner: Optional[nn.Module] = None,
                 org_dim: int = 258,
                 ) -> None:
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(in_dim, out_dim - org_dim),
            nn.ReLU(),
            )
        self.inner = inner

    def forward(self, x: Tensor) -> Tensor:
        if self.inner is None:
            return cat([self.linear(x), x], dim=1)
        return cat([self.linear(self.inner(x)), x], dim=1)


if __name__ == "__main__":
    from torch import randn

    dim = 256
    net = IMNetBlock(
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
        )
    y = net(randn(16, dim + 2))
    print(y.shape)
