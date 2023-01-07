import random
from typing import Tuple

from torch import Tensor
from torchvision.transforms import functional as tf

from .interface import Interface


class RandomFlip(Interface):
    def forward(self, x: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        bw, rgb = x
        if random.random() > 0.5:
            return tf.hflip(bw), tf.hflip(rgb)
        return bw, rgb
