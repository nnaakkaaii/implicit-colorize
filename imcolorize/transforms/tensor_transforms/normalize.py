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


if __name__ == "__main__":
    # python3 -m imcolorize.transforms.tensor_transforms.normalize
    from ...datasets.stl10 import STL10
    s = STL10(pil_transforms=[], tensor_transforms=[Normalize()])
    b, r = next(iter(s))
    print(b.shape, r.shape)
