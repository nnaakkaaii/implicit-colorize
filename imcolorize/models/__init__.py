from typing import Dict, Type

from .autoencoder import AutoEncoder
from .imnet import IMNet
from .interface import Interface
from .unet import UNet

models: Dict[str, Type[Interface]] = {
    "autoencoder": AutoEncoder,
    "imnet": IMNet,
    "unet": UNet,
}
