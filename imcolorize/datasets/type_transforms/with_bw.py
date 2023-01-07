from typing import Tuple

from PIL.Image import Image

from ...transforms.interface import Interface


class WithBW(Interface):
    def forward(self, x: Tuple[Image, int]) -> Tuple[Image, Image]:
        rgb, _ = x
        rgb: Image
        bw = rgb.convert('1')
        return bw, rgb
