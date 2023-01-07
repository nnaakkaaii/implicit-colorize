from torch import nn, Tensor

from .resnet_encoder import ResNetEncoder
from .imnet_decoder import IMNetDecoder


class IMNetGenerator(nn.Module):
    def __init__(self,
                 img_size: int = 96,
                 ) -> None:
        super().__init__()
        
        self.encoder = ResNetEncoder()
        self.decoder = IMNetDecoder(img_size)

    def forward(self, x: Tensor) -> Tensor:
        h = self.encoder(x)
        y = self.decoder(h)
        return y


if __name__ == "__main__":
    # python3 -m imcolorize.networks.imnet_generator
    from torch import randn
    
    net = IMNetGenerator()
    pred = net(randn(16, 1, 96, 96))
    print(pred.shape)
