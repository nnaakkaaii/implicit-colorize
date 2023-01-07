from typing import Tuple

from torchvision import transforms
from torch import Tensor
from torchvision.transforms import functional as tf

from .interface import Interface


class RandomCrop(Interface):
    def forward(self, x: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        bw, rgb = x
        w_bw, h_bw = bw.size
        w_rgb, h_rgb = rgb.size
        assert w_bw == h_bw == w_rgb == h_rgb
        p = transforms.RandomCrop.get_params(bw, output_size=(w_bw, h_bw))
        return tf.crop(bw, *p), tf.crop(rgb, *p)
