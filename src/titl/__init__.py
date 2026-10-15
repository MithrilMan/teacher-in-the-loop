"""Teacher in the Loop: 8M-parameter single-voice students of Kokoro-82M, in English and Italian."""
from .voice import SAMPLE_RATE, VOICES, Voice, write_wav

__all__ = ["SAMPLE_RATE", "VOICES", "Voice", "write_wav"]
__version__ = "0.1.0"
