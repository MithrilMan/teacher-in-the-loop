"""titl --voice if_sara "Buongiorno a tutti." -o buongiorno.wav"""
import argparse
import sys

from . import SAMPLE_RATE, VOICES, Voice, write_wav


def main(argv=None):
    ap = argparse.ArgumentParser(prog="titl", description="Speak text with a distilled voice and save a WAV file.")
    ap.add_argument("text", nargs="?", help="text to speak (read from stdin when omitted)")
    ap.add_argument("-v", "--voice", default="af_heart", choices=sorted(VOICES))
    ap.add_argument("-o", "--out", default="out.wav")
    ap.add_argument("--speed", type=float, default=1.0, help="above 1 speaks faster")
    ap.add_argument("--fp32", action="store_true", help="use the float32 graph instead of int8")
    ap.add_argument("--model", help="a local ONNX graph instead of the released one (language from --voice)")
    ap.add_argument("--phonemes", action="store_true", help="print the phonemes and exit")
    a = ap.parse_args(argv)
    text = a.text if a.text is not None else sys.stdin.read()
    if a.phonemes:
        from .phonemes import Phonemizer

        sys.stdout.reconfigure(encoding="utf-8")
        print("\n".join(Phonemizer(VOICES[a.voice]).chunks(text)))
        return
    voice = Voice(a.model, VOICES[a.voice]) if a.model else Voice.from_pretrained(a.voice, quantized=not a.fp32)
    audio = voice(text, speed=a.speed)
    write_wav(a.out, audio)
    print(f"wrote {a.out} ({len(audio) / SAMPLE_RATE:.1f} s)")


if __name__ == "__main__":
    main()
