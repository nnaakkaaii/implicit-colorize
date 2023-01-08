from torch import Tensor, nn


class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = tuple(args)

    def forward(self, x: Tensor) -> Tensor:
        return x.view(-1, *self.shape)
