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


if __name__ == "__main__":
    # python3 -m imcolorize.transforms.tensor_transforms.random_flip
    from ...datasets.stl10 import STL10
    s = STL10(pil_transforms=[], tensor_transforms=[RandomFlip()])
    x = next(iter(s))
    print(x[0].shape, x[1].shape)
