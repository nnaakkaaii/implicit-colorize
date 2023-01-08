from torch import Tensor, nn

from ..networks.cnn_decoder import CNNDecoder
from ..networks.cnn_encoder import CNNEncoder
from .encoder_decoder_wrapper import EncoderDecoderWrapper


class AutoEncoder(EncoderDecoderWrapper):
    encoder_filename = "net_resnet_encoder.pth"
    decoder_filename = "net_cnn_decoder.pth"

    def __init__(self) -> None:
        super().__init__()
        self.encoder = CNNEncoder()
        self.decoder = CNNDecoder()
        self.criterion = nn.MSELoss()

    def loss(self, t: Tensor) -> Tensor:
        return self.criterion(self.y, self.x.to(self.device))

    def backward(self, t: Tensor) -> float:
        self.optimizer.zero_grad()
        loss = self.loss(t)
        loss.backward()
        self.optimizer.step()
        return loss.item()
