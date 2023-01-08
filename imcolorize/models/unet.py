import os
from pathlib import Path

import torch
from torch import Tensor, nn, optim
from torch.nn import DataParallel

from ..networks.unet_generator import UNetGenerator
from .interface import Interface


class UNet(Interface):
    net_filename = "net_unet.pth"
    net: nn.Module

    def __init__(self) -> None:
        self.net = UNetGenerator()
        self.criterion = nn.MSELoss()

        self.y = torch.zeros(0)

    def to(self, device: str) -> None:
        if device == "cuda:0":
            self.net = DataParallel(self.net)
            torch.backends.cudnn.benchmark = True
        self.net.to(device)
        self.device = device

    def load_state_dict(self, save_dir: Path) -> None:
        net_path = save_dir / self.net_filename
        if os.path.isfile(net_path):
            self.net.load_state_dict(torch.load(net_path))

    def save_state_dict(self, save_dir: Path) -> None:
        net_path = save_dir / self.net_filename
        if isinstance(self.net, DataParallel):
            torch.save(self.net.module.state_dict(), net_path)
        else:
            torch.save(self.net.state_dict(), net_path)

    def forward(self, x: Tensor) -> Tensor:
        self.y = self.net(x)
        return self.y

    def set_optim(self, **kwargs) -> None:
        assert "lr" in kwargs
        self.optimizer = optim.Adam(self.net.parameters(), lr=kwargs["lr"])

    def backward(self, t: Tensor) -> float:
        self.optimizer.zero_grad()
        loss = self.criterion(self.y, t.to(self.device))
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train(self) -> None:
        self.net.train()

    def eval(self) -> None:
        self.net.eval()
