import csv
import os
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from imcolorize.datasets.stl10 import STL10
from imcolorize.models import models
from imcolorize.transforms.tensor_transforms.interface import Interface
from imcolorize.transforms.tensor_transforms.normalize import Normalize
from imcolorize.transforms.tensor_transforms.random_affine import RandomAffine
from imcolorize.transforms.tensor_transforms.random_crop import RandomCrop
from imcolorize.transforms.tensor_transforms.random_flip import RandomFlip


def run(lr: float,
        encoder_lr: float,
        decoder_lr: float,
        step_size: int,
        gamma: float,
        batch_size: int,
        num_epochs: int,
        save_dir: Path,
        model_name: str,
        use_random_crop: bool,
        use_random_flip: bool,
        use_random_affine: bool,
        ) -> float:
    os.makedirs(save_dir, exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    assert model_name in models
    model = models[model_name]()
    model.load_state_dict(save_dir)
    model.to(device)

    tensor_transformers: List[Interface] = []
    if use_random_flip:
        tensor_transformers.append(RandomFlip())
    if use_random_crop:
        tensor_transformers.append(RandomCrop())
    if use_random_affine:
        tensor_transformers.append(RandomAffine())
    tensor_transformers.append(Normalize())

    train_set = STL10(pil_transforms=[],
                      tensor_transforms=tensor_transformers,
                      phase="train",
                      )
    test_set = STL10(pil_transforms=[],
                     tensor_transforms=[Normalize()],
                     phase="test",
                     )

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              )
    test_loader = DataLoader(test_set,
                             batch_size=batch_size,
                             shuffle=False,
                             )

    model.set_optim(
        lr=lr,
        encoder_lr=encoder_lr,
        decoder_lr=decoder_lr,
        )
    scheduler = torch.optim.lr_scheduler.StepLR(
        model.optimizer,
        step_size=step_size,
        gamma=gamma,
        )

    with open(save_dir / "result.csv", "a") as f:
        csv.writer(f).writerow([
            "dt",
            "epoch",
            "train_loss",
            "test_loss",
            ])

    best_test_loss = 1e8

    for epoch in range(1, num_epochs + 1):
        model.train()

        train_losses = []
        for bw, rgb in tqdm(train_loader):
            model.forward(bw)
            loss = model.backward(rgb)
            train_losses.append(loss)
            break

        scheduler.step()

        model.eval()

        test_losses = []
        with torch.no_grad():
            for bw, rgb in tqdm(test_loader):
                model.forward(bw)
                loss = model.loss(rgb).item()
                test_losses.append(loss)

        train_loss = float(np.mean(train_losses))
        test_loss = float(np.mean(test_losses))

        best_test_loss = min(test_loss, best_test_loss)

        print(f"[Epoch {epoch:04}]: "
              f"train loss {train_loss:.3f} / "
              f"test loss {test_loss:.3f} / "
              f"lr {scheduler.get_last_lr()[0]:.5f}")

        with open(save_dir / "result.csv", "a") as f:
            csv.writer(f).writerow([
                datetime.now(),
                epoch,
                train_loss,
                test_loss,
                ])

        if best_test_loss == test_loss:
            model.save_state_dict(save_dir)

    return best_test_loss


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--lr",
                        type=float,
                        default=0.01,
                        )
    parser.add_argument("--encoder_lr",
                        type=float,
                        default=0.01,
                        )
    parser.add_argument("--decoder_lr",
                        type=float,
                        default=0.01,
                        )
    parser.add_argument("--step_size",
                        type=int,
                        default=10,
                        )
    parser.add_argument("--gamma",
                        type=float,
                        default=0.5,
                        )
    parser.add_argument("--batch_size",
                        type=int,
                        default=16,
                        )
    parser.add_argument("--num_epochs",
                        type=int,
                        default=50,
                        )
    parser.add_argument("--save_dir",
                        type=str,
                        default="results/test01",
                        )
    parser.add_argument("--model_name",
                        type=str,
                        choices=list(models.keys()),
                        default="imnet",
                        )
    parser.add_argument("--use_random_crop",
                        action="store_true",
                        )
    parser.add_argument("--use_random_flip",
                        action="store_true",
                        )
    parser.add_argument("--use_random_affine",
                        action="store_true",
                        )
    args = parser.parse_args()

    best = run(args.lr,
               args.encoder_lr,
               args.decoder_lr,
               args.step_size,
               args.gamma,
               args.batch_size,
               args.num_epochs,
               Path(args.save_dir),
               args.model_name,
               args.use_random_crop,
               args.use_random_flip,
               args.use_random_affine,
               )
