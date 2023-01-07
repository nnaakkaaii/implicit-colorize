from .imnet_generator import IMNetGenerator
from .unet_generator import UNetGenerator

networks = {
    "unet": UNetGenerator,
    "imnet": IMNetGenerator,
}
