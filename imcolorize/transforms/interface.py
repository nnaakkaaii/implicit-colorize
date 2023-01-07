from abc import ABCMeta, abstractmethod
from typing import Any, Tuple


class Interface(metaclass=ABCMeta):
    @abstractmethod
    def forward(self, x: Any) -> Tuple[Any, Any]:
        pass

    def __call__(self, x: Any) -> Tuple[Any, Any]:
        return self.forward(x)
