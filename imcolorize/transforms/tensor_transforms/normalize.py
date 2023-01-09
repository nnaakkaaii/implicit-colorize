from typing import Tuple

from torch import Tensor
from torchvision import transforms

from .interface import Interface


class Normalize(Interface):
    BW_MEAN = (0.5,)
    BW_STD = (0.5,)
    RGB_MEAN = (0.5, 0.5, 0.5)
    RGB_STD = (0.5, 0.5, 0.5)

    def forward(self, x: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        bw, rgb = x
        bw_normalize = transforms.Normalize(self.BW_MEAN, self.BW_STD)
        rgb_normalize = transforms.Normalize(self.RGB_MEAN, self.RGB_STD)
        return bw_normalize(bw), rgb_normalize(rgb)

    def backward(self, x: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        bw, rgb = x
        bw_normalize = transforms.Compose([
            transforms.Normalize((0,), tuple(1/i for i in self.BW_STD)),
            transforms.Normalize(tuple(-i for i in self.BW_MEAN), (1,)),
            ])
        rgb_normalize = transforms.Compose([
            transforms.Normalize((0, 0, 0), tuple(1/i for i in self.RGB_STD)),
            transforms.Normalize(tuple(-i for i in self.RGB_MEAN), (1, 1, 1)),
            ])
        return bw_normalize(bw), rgb_normalize(rgb)


if __name__ == "__main__":
    # python3 -m imcolorize.transforms.tensor_transforms.normalize
    from ...datasets.stl10 import STL10
    s = STL10(pil_transforms=[], tensor_transforms=[])
    b, r = next(iter(s))
    print(b.shape, r.shape)
    print(b.min(), b.max(), r.min(), r.max())
    b, r = Normalize().forward((b, r))
    print(b.shape, r.shape)
    print(b.min(), b.max(), r.min(), r.max())
    b, r = Normalize().backward((b, r))
    print(b.shape, r.shape)
    print(b.min(), b.max(), r.min(), r.max())

    from ...transforms.pil_transforms.rgb2ycbcr import RGB2YCbCr
    s = STL10(pil_transforms=[RGB2YCbCr()], tensor_transforms=[])
    b, r = next(iter(s))
    print(b.shape, r.shape)
    print(b.min(), b.max(), r.min(), r.max())
    b, r = Normalize().forward((b, r))
    print(b.shape, r.shape)
    print(b.min(), b.max(), r.min(), r.max())
    b, r = Normalize().backward((b, r))
    print(b.shape, r.shape)
    print(b.min(), b.max(), r.min(), r.max())
