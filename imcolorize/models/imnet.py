import os
from pathlib import Path

import torch
from torch import Tensor, nn, optim
from torch.nn import DataParallel

from ..networks.cnn_encoder import CNNEncoder
from ..networks.imnet_decoder import IMNetDecoder
from .encoder_decoder_wrapper import EncoderDecoderWrapper


class IMNet(EncoderDecoderWrapper):
    """IMNet freezes encoder & trains decoder"""
    encoder_filename = "net_resnet_encoder.pth"
    decoder_filename = "net_imnet_decoder.pth"

    def __init__(self) -> None:
        super().__init__()
        self.encoder = CNNEncoder()
        self.decoder = IMNetDecoder()
        self.criterion = nn.MSELoss()

        self.encoder.eval()

    def load_state_dict(self, save_dir: Path) -> None:
        net_encoder_path = save_dir / self.encoder_filename
        assert os.path.isfile(net_encoder_path)
        self.encoder.load_state_dict(torch.load(net_encoder_path))

        net_decoder_path = save_dir / self.decoder_filename
        if os.path.isfile(net_decoder_path):
            self.decoder.load_state_dict(torch.load(net_decoder_path))

    def save_state_dict(self, save_dir: Path) -> None:
        net_decoder_path = save_dir / self.decoder_filename
        if isinstance(self.decoder, DataParallel):
            torch.save(self.decoder.module.state_dict(), net_decoder_path)
        else:
            torch.save(self.decoder.state_dict(), net_decoder_path)

    def loss(self, t: Tensor) -> Tensor:
        if self.y.size(1) == 2:
            return self.criterion(self.y, t[:, 1:, :, :].to(self.device))
        return self.criterion(self.y, t.to(self.device))

    def forward(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            h = self.encoder(x.to(self.device))
        y = self.decoder(h)

        self.x = x
        self.y = y

        return y

    def backward(self, t: Tensor) -> float:
        self.optimizer.zero_grad()
        loss = self.loss(t)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def set_optim(self, **kwargs) -> None:
        assert "decoder_lr" in kwargs
        self.optimizer = optim.Adam(
            self.decoder.parameters(),
            lr=kwargs["decoder_lr"],
            )

    def train(self) -> None:
        self.decoder.train()

    def eval(self) -> None:
        self.decoder.eval()
