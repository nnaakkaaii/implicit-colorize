from typing import Optional, Tuple, List

from torchvision import datasets
from torch import Tensor

from .interface import Interface as DatasetInterface
from ..transforms.pil_transforms.interface import Interface as PilTransformsInterface
from ..transforms.tensor_transforms.interface import Interface as TensorTransformsInterface


class STL10(DatasetInterface):
    DEFAULT_SAVE_DIR = "data/stl10"
    PHASES = {
        "train": "train+unlabeled",
        "test": "test",
    }
    
    def __init__(self,
                 pil_transforms: List[PilTransformsInterface],
                 tensor_transforms: List[TensorTransformsInterface],
                 phase: str = "train",
                 save_dir: Optional[str] = None,
                 ) -> None:
        assert phase in self.PHASES
        super().__init__()
        
        if save_dir is None:
            save_dir = self.DEFAULT_SAVE_DIR

        self.dataset = datasets.STL10(
            root=save_dir,
            split=self.PHASES[phase],
            download=True,
            )
        self.transform = self.transforms(pil_transforms,
                                         tensor_transforms)

    def __getitem__(self, item: int) -> Tuple[Tensor, Tensor]:
        bw, rgb = self.transform(self.dataset[item])
        return bw, rgb

    def __len__(self) -> int:
        return len(self.dataset)


if __name__ == '__main__':
    s = STL10(pil_transforms=[], tensor_transforms=[])
    print(len(s))
    b, r = next(iter(s))
    print(type(b), type(r))
