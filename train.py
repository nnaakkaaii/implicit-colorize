import csv
import os
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from imcolorize.datasets.stl10 import STL10
from imcolorize.networks import networks
from imcolorize.transforms.tensor_transforms.interface import Interface
from imcolorize.transforms.tensor_transforms.normalize import Normalize
from imcolorize.transforms.tensor_transforms.random_affine import RandomAffine
from imcolorize.transforms.tensor_transforms.random_crop import RandomCrop
from imcolorize.transforms.tensor_transforms.random_flip import RandomFlip


def run(lr: float,
        step_size: int,
        gamma: float,
        batch_size: int,
        num_epochs: int,
        save_dir: Path,
        network_name: str,
        use_random_crop: bool,
        use_random_flip: bool,
        use_random_affine: bool,
        ) -> float:
    os.makedirs(save_dir, exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    assert network_name in networks
    net = networks[network_name]()
    if device == "cuda:0":
        net = torch.nn.DataParallel(net)
        net.to(device)
        torch.backends.cudnn.benchmark = True

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

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
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
        net.train()

        train_losses = []
        for bw, rgb in tqdm(train_loader):
            pred = net(bw.to(device))
            optimizer.zero_grad()
            loss = criterion(pred, rgb.to(device))
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            break

        scheduler.step()

        net.eval()

        test_losses = []
        with torch.no_grad():
            for bw, rgb in tqdm(test_loader):
                pred = net(bw.to(device))
                loss = criterion(pred, rgb.to(device))
                test_losses.append(loss.item())
                break

        train_loss = np.mean(train_losses)
        test_loss = np.mean(test_losses)

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
            save_path = save_dir / f"net_{network_name}.pth"
            if isinstance(net, torch.nn.DataParallel):
                torch.save(net.module.state_dict(), save_path)
            else:
                torch.save(net.state_dict(), save_path)

    return best_test_loss


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--lr",
                        type=float,
                        default=0.01,
                        )
    parser.add_argument("--step_size",
                        type=int,
                        default=20,
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
    parser.add_argument("--network_name",
                        type=str,
                        choices=list(networks.keys()),
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
               args.step_size,
               args.gamma,
               args.batch_size,
               args.num_epochs,
               Path(args.save_dir),
               args.network_name,
               args.use_random_crop,
               args.use_random_flip,
               args.use_random_affine,
               )
