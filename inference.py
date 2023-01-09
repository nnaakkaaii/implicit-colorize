from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

from imcolorize.datasets.stl10 import STL10
from imcolorize.models import models
from imcolorize.transforms.tensor_transforms.normalize import Normalize


def run(save_dir: Path,
        model_name: str,
        num_samples: int,
        ) -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    assert model_name in models
    model = models[model_name]()

    model.load_state_dict(save_dir)
    model.to(device)

    norm = Normalize()

    test_set = STL10(pil_transforms=[],
                     tensor_transforms=[norm],
                     phase="test",
                     )
    test_loader = DataLoader(test_set,
                             batch_size=1,
                             shuffle=False,
                             )

    model.eval()
    with torch.no_grad():
        for i, (bw, rgb) in enumerate(tqdm(test_loader)):
            if i % (len(test_loader) / num_samples) != 0:
                continue
            pred = model.forward(bw).cpu().detach().clone()

            if pred.size(1) == 1:
                pred, rgb = norm.backward((pred, rgb))
            elif pred.size(1) == 3:
                _, rgb = norm.backward((bw, rgb))
                _, pred = norm.backward((bw, pred))
            else:
                raise NotImplementedError

            rgb_img = to_pil_image(rgb[0])
            pred_img = to_pil_image(pred[0])

            rgb_img.save(save_dir / f"{i}__real.jpg")
            pred_img.save(save_dir / f"{i}_{model_name}.jpg")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir",
                        type=str,
                        default="results/test01",
                        )
    parser.add_argument("--model_name",
                        type=str,
                        choices=list(models.keys()),
                        default="imnet",
                        )
    parser.add_argument("--num_samples",
                        type=int,
                        default=10)
    args = parser.parse_args()

    run(Path(args.save_dir),
        args.model_name,
        args.num_samples,
        )
