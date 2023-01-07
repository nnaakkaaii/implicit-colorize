from typing import Tuple

from torch import Tensor
from torchvision import transforms
from torchvision.transforms import functional as tf

from .interface import Interface


class RandomCrop(Interface):
    def forward(self, x: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        bw, rgb = x
        _, w_bw, h_bw = bw.size()
        _, w_rgb, h_rgb = rgb.size()
        assert w_bw == h_bw == w_rgb == h_rgb
        p = transforms.RandomCrop.get_params(bw, output_size=(w_bw, h_bw))
        return tf.crop(bw, *p), tf.crop(rgb, *p)


if __name__ == "__main__":
    # python3 -m imcolorize.transforms.tensor_transforms.random_crop
    from ...datasets.stl10 import STL10
    s = STL10(pil_transforms=[], tensor_transforms=[RandomCrop()])
    b, r = next(iter(s))
    print(b.shape, r.shape)
