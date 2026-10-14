"""Inference for two 小光 (xiǎo guāng) wake word models."""
from .openwakeword_detector import OpenWakeWordDetector
from .sherpa_detector import SherpaKwsDetector

__all__ = ["OpenWakeWordDetector", "SherpaKwsDetector"]
__version__ = "0.1.0"
