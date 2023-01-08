from torch import Tensor, nn

from ..networks.cnn_decoder import CNNDecoder
from ..networks.resnet_encoder import ResNetEncoder
from .encoder_decoder_wrapper import EncoderDecoderWrapper


class AutoEncoder(EncoderDecoderWrapper):
    encoder_filename = "net_resnet_encoder.pth"
    decoder_filename = "net_cnn_decoder.pth"

    def __init__(self) -> None:
        super().__init__()
        self.encoder = ResNetEncoder()
        self.decoder = CNNDecoder()
        self.criterion = nn.MSELoss()

    def backward(self, t: Tensor) -> float:
        self.optimizer.zero_grad()
        loss = self.criterion(self.y, self.x.to(self.device))
        loss.backward()
        self.optimizer.step()
        return loss.item()
