import os
from abc import ABCMeta
from pathlib import Path

import torch
from torch import Tensor, nn, optim
from torch.nn import DataParallel

from .interface import Interface


class EncoderDecoderWrapper(Interface, metaclass=ABCMeta):
    encoder_filename: str
    decoder_filename: str
    encoder: nn.Module
    decoder: nn.Module

    def __init__(self) -> None:
        self.x = torch.zeros(0)
        self.y = torch.zeros(0)

    def to(self, device: str) -> None:
        if device == "cuda:0":
            self.encoder = DataParallel(self.encoder)
            self.decoder = DataParallel(self.decoder)
            torch.backends.cudnn.benchmark = True
        self.encoder.to(device)
        self.decoder.to(device)
        self.device = device

    def load_state_dict(self, save_dir: Path) -> None:
        net_decoder_path = save_dir / self.decoder_filename
        if os.path.isfile(net_decoder_path):
            self.decoder.load_state_dict(torch.load(net_decoder_path))
        net_encoder_path = save_dir / self.encoder_filename
        if os.path.isfile(net_encoder_path):
            self.encoder.load_state_dict(torch.load(net_encoder_path))

    def save_state_dict(self, save_dir: Path) -> None:
        net_decoder_path = save_dir / self.decoder_filename
        if isinstance(self.decoder, DataParallel):
            torch.save(self.decoder.module.state_dict(), net_decoder_path)
        else:
            torch.save(self.decoder.state_dict(), net_decoder_path)
        net_encoder_path = save_dir / self.encoder_filename
        if isinstance(self.encoder, DataParallel):
            torch.save(self.encoder.module.state_dict(), net_encoder_path)
        else:
            torch.save(self.encoder.state_dict(), net_encoder_path)

    def forward(self, x: Tensor) -> Tensor:
        h = self.encoder(x.to(self.device))
        y = self.decoder(h)

        self.x = x
        self.y = y

        return y

    def set_optim(self, **kwargs) -> None:
        assert "encoder_lr" in kwargs
        assert "decoder_lr" in kwargs
        self.optimizer = optim.Adam([
            {"params": self.encoder.parameters(), "lr": kwargs["encoder_lr"]},
            {"params": self.decoder.parameters(), "lr": kwargs["decoder_lr"]},
            ])

    def train(self) -> None:
        self.encoder.train()
        self.decoder.train()

    def eval(self) -> None:
        self.encoder.eval()
        self.decoder.eval()
