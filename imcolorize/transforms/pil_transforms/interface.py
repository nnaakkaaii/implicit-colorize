from abc import abstractmethod
from typing import Tuple

from PIL.Image import Image

from ..interface import Interface as TransformsInterface


class Interface(TransformsInterface):
    @abstractmethod
    def forward(self, x: Tuple[Image, Image]) -> Tuple[Image, Image]:
        pass

    def __call__(self, x: Tuple[Image, Image]) -> Tuple[Image, Image]:
        return self.forward(x)
