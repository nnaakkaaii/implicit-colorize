from typing import Tuple

from torch import Tensor
from torchvision import transforms

from .interface import Interface


class Normalize(Interface):
    def forward(self, x: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        bw, rgb = x
        bw_normalize = transforms.Normalize((0.5,), (0.5,))
        rgb_normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        return bw_normalize(bw), rgb_normalize(rgb)
