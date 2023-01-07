from typing import Optional, Tuple

from torch import Tensor
from torchvision import transforms
from torchvision.transforms import functional as tf

from .interface import Interface


class RandomAffine(Interface):
    DEFAULT_DEGREE = 15
    DEFAULT_TRANSLATE_X = 0.1
    DEFAULT_TRANSLATE_Y = 0.1
    DEFAULT_SCALE_MIN = 0.8
    DEFAULT_SCALE_MAX = 1.2

    def __init__(self,
                 degree: Optional[int] = None,
                 translate_x: Optional[float] = None,
                 translate_y: Optional[float] = None,
                 scale_min: Optional[float] = None,
                 scale_max: Optional[float] = None,
                 ) -> None:
        self.degree = [-self.DEFAULT_DEGREE, self.DEFAULT_DEGREE]
        self.translate = [self.DEFAULT_TRANSLATE_X, self.DEFAULT_TRANSLATE_Y]
        self.scale = [self.DEFAULT_SCALE_MIN, self.DEFAULT_SCALE_MAX]

        if degree is not None:
            self.degree = [-degree, degree]
        if translate_x is not None:
            self.translate[0] = translate_x
        if translate_y is not None:
            self.translate[1] = translate_y
        if scale_min is not None:
            self.scale[0] = scale_min
        if scale_max is not None:
            self.scale[1] = scale_max

    def forward(self, x: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        bw, rgb = x
        _, w_bw, h_bw = bw.size()
        _, w_rgb, h_rgb = rgb.size()
        assert w_bw == h_bw == w_rgb == h_rgb
        p = transforms.RandomAffine.get_params(
            degrees=self.degree,
            translate=self.translate,
            scale_ranges=self.scale,
            shears=None,
            img_size=[w_bw, h_bw]
            )
        return tf.affine(bw, *p), tf.affine(rgb, *p)


if __name__ == "__main__":
    # python3 -m imcolorize.transforms.tensor_transforms.random_affine
    from ...datasets.stl10 import STL10
    s = STL10(pil_transforms=[], tensor_transforms=[RandomAffine()])
    b, r = next(iter(s))
    print(b.shape, r.shape)
