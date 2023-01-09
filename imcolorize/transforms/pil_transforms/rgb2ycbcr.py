from typing import Tuple

from PIL.Image import Image

from .interface import Interface


class RGB2YCbCr(Interface):
    def forward(self, x: Tuple[Image, Image]) -> Tuple[Image, Image]:
        bw, rgb = x
        ycrcb = rgb.convert("YCbCr")
        return bw, ycrcb

    def backward(self, x: Tuple[Image, Image]) -> Tuple[Image, Image]:
        bw, ycrcb = x
        rgb = ycrcb.convert("RGB")
        return bw, rgb


if __name__ == "__main__":
    # python3 -m imcolorize.transforms.pil_transforms.rgb2ycrcb
    from ...datasets.stl10 import STL10
    s = STL10(pil_transforms=[RGB2YCbCr()], tensor_transforms=[])
    b, r = next(iter(s))
    print(b.shape, r.shape)
    print((b - r[:1]).sum())
    print((b - r[1:2]).sum())
    print((b - r[2:]).sum())
