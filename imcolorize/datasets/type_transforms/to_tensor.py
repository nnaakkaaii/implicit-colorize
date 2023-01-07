from typing import Tuple

from PIL.Image import Image
from torch import Tensor
from torchvision.transforms.functional import to_tensor

from ...transforms.interface import Interface


class ToTensor(Interface):
    def forward(self, x: Tuple[Image, Image]) -> Tuple[Tensor, Tensor]:
        bw, rgb = x
        t_bw = to_tensor(bw)
        t_rgb = to_tensor(rgb)
        return t_bw, t_rgb
