from abc import abstractmethod
from typing import Tuple

from torch import Tensor

from ..interface import Interface as TransformsInterface


class Interface(TransformsInterface):
    @abstractmethod
    def forward(self, x: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        pass

    def __call__(self, x: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        return self.forward(x)
