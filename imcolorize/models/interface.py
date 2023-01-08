from abc import ABCMeta, abstractmethod
from pathlib import Path

from torch import Tensor, nn, optim


class Interface(metaclass=ABCMeta):
    criterion: nn.Module
    optimizer: optim.Optimizer

    device = "cpu"

    @abstractmethod
    def to(self, device: str) -> None:
        pass

    @abstractmethod
    def load_state_dict(self, save_dir: Path) -> None:
        pass

    @abstractmethod
    def save_state_dict(self, save_dir: Path) -> None:
        pass

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        pass

    @abstractmethod
    def set_optim(self, **kwargs) -> None:
        pass

    @abstractmethod
    def loss(self, t: Tensor) -> Tensor:
        pass

    @abstractmethod
    def backward(self, t: Tensor) -> float:
        pass

    @abstractmethod
    def train(self) -> None:
        pass

    @abstractmethod
    def eval(self) -> None:
        pass
