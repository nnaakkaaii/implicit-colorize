from torch import Tensor, nn

from .cnn_decoder import CNNDecoder
from .cnn_encoder import CNNEncoder


class CNNAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            CNNEncoder(),
            CNNDecoder(),
            )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


if __name__ == "__main__":
    # python3 -m imcolorize.networks.cnn_autoencoder
    from torch import randn

    net = CNNAutoEncoder()
    y = net(randn(16, 1, 96, 96))
    print(y.shape)
