from abc import ABCMeta, abstractmethod
from typing import List, Tuple

from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from ..transforms.pil_transforms.interface import \
    Interface as PilTransformsInterface
from ..transforms.tensor_transforms.interface import \
    Interface as TensorTransformsInterface
from .type_transforms.to_tensor import ToTensor
from .type_transforms.with_bw import WithBW


class Interface(Dataset, metaclass=ABCMeta):
    @abstractmethod
    def __getitem__(self, item: int) -> Tuple[Tensor, Tensor]:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @staticmethod
    def transforms(pil_transforms: List[PilTransformsInterface],
                   tensor_transforms: List[TensorTransformsInterface],
                   ):
        return Compose([WithBW()]
                       + pil_transforms
                       + [ToTensor()]
                       + tensor_transforms)
