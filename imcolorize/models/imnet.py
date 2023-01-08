from torch import Tensor, nn

from ..networks.imnet_decoder import IMNetDecoder
from ..networks.resnet_encoder import ResNetEncoder
from .encoder_decoder_wrapper import EncoderDecoderWrapper


class IMNet(EncoderDecoderWrapper):
    encoder_filename = "net_resnet_encoder.pth"
    decoder_filename = "net_imnet_decoder.pth"

    def __init__(self) -> None:
        super().__init__()
        self.encoder = ResNetEncoder()
        self.decoder = IMNetDecoder()
        self.criterion = nn.MSELoss()

    def backward(self, t: Tensor) -> float:
        self.optimizer.zero_grad()
        loss = self.criterion(self.y, t.to(self.device))
        loss.backward()
        self.optimizer.step()
        return loss.item()
