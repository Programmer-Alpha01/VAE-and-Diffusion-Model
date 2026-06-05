# DDPM/__init__.py
# This file makes DDPM a package.
# You can leave it completely empty, or add optional exports / version info.

__version__ = "0.1.0"  # optional

# Optional: expose main classes for easier imports
from .Unet.unet import Unet
from .DiffusionForwardProcess import DiffusionForwardProcess
from .DiffusionReverseProcess import DiffusionReverseProcess