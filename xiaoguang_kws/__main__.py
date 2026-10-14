"""Detect 小光 in WAV files or from the microphone with the openWakeWord and sherpa-onnx models.

    xiaoguang-kws clip.wav [more.wav ...]
    xiaoguang-kws --mic                      # requires the `mic` extra (sounddevice)
"""
import argparse
import math
import queue
import time

import numpy as np

SAMPLE_RATE = 16000
CHUNK = 1600  # 100 ms
REFRACTORY_SECONDS = 1.0  # openWakeWord scores stay above the threshold for several steps of one utterance


def load_wav(path: str) -> np.ndarray:
    import soundfile as sf
    from scipy.signal import resample_poly
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sample_rate != SAMPLE_RATE:
        divisor = math.gcd(sample_rate, SAMPLE_RATE)
        audio = resample_poly(audio, SAMPLE_RATE // divisor, sample_rate // divisor)
    return (np.clip(audio, -1, 1) * 32767).astype(np.int16)


class Spotter:
    """Runs the selected detectors on a stream and turns their output into timed detections."""

    def __init__(self, args):
        self.oww = self.sherpa = None
        if "openwakeword" in args.models:
            from .openwakeword_detector import OpenWakeWordDetector
            self.oww = OpenWakeWordDetector(threshold=args.oww_threshold, local_dir=args.local_dir)
        if "sherpa" in args.models:
            from .sherpa_detector import SherpaKwsDetector
            self.sherpa = SherpaKwsDetector(threshold=args.sherpa_threshold, boost=args.sherpa_boost,
                                            keywords_file=args.keywords, local_dir=args.local_dir, int8=args.int8)
        self.reset()

    def reset(self):
        for detector in (self.oww, self.sherpa):
            if detector:
                detector.reset()
        self.samples = 0
        self.oww_steps = 0
        self.last_oww = -math.inf

    def feed(self, pcm16: np.ndarray):
        """Returns [(seconds, model, keyword, score or None)] for the detections ending in this chunk."""
        events = []
        self.samples += len(pcm16)
        if self.oww:
            for score in self.oww.process(pcm16):
                self.oww_steps += 1
                t = self.oww_steps * 0.08
                if score >= self.oww.threshold and t - self.last_oww >= REFRACTORY_SECONDS:
                    self.last_oww = t
                    events.append((t, "openWakeWord", "小光", score))
        if self.sherpa:
            events += [(self.samples / SAMPLE_RATE, "sherpa-onnx", keyword, None) for keyword in self.sherpa.process(pcm16)]
        return sorted(events)


def print_event(event, prefix=""):
    t, model, keyword, score = event
    print(f"{prefix}{t:8.2f} s  {model:<13} {keyword}" + (f"  (score {score:.2f})" if score is not None else ""), flush=True)


def run_files(spotter, paths):
    for path in paths:
        spotter.reset()
        # Trailing silence lets a wake word at the very end of the file complete
        audio = np.concatenate([load_wav(path), np.zeros(SAMPLE_RATE, dtype=np.int16)])
        print(path)
        events = [e for i in range(0, len(audio), CHUNK) for e in spotter.feed(audio[i:i + CHUNK])]
        for event in events:
            print_event(event, "  ")
        if not events:
            print("  no detection")


def run_microphone(spotter):
    import sounddevice as sd
    blocks = queue.Queue()
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16", blocksize=CHUNK,
                        callback=lambda data, frames, t, status: blocks.put(data[:, 0].copy())):
        print("Listening; say 小光 (Ctrl+C to stop)", flush=True)
        start = time.monotonic()
        try:
            while True:
                for event in spotter.feed(blocks.get()):
                    print_event((time.monotonic() - start,) + event[1:])
        except KeyboardInterrupt:
            pass


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("wavs", nargs="*", help="audio files (any sample rate; converted to 16 kHz mono)")
    parser.add_argument("--mic", action="store_true", help="listen to the default microphone")
    parser.add_argument("--models", nargs="+", choices=["openwakeword", "sherpa"], default=["openwakeword", "sherpa"])
    parser.add_argument("--oww-threshold", type=float, default=0.65, help="openWakeWord score threshold")
    parser.add_argument("--sherpa-threshold", type=float, default=0.25, help="sherpa-onnx keyword threshold")
    parser.add_argument("--sherpa-boost", type=float, default=0.2,
                        help="sherpa-onnx keyword boosting score; larger values trigger more often, also on near-misses")
    parser.add_argument("--keywords", help="sherpa-onnx keywords file (default: the one shipped with the model)")
    parser.add_argument("--int8", action="store_true", help="use the int8-quantized sherpa-onnx model")
    parser.add_argument("--local-dir", help="directory containing the model folders, instead of downloading them")
    args = parser.parse_args()
    if not args.wavs and not args.mic:
        parser.error("give WAV files or --mic")

    spotter = Spotter(args)
    if args.wavs:
        run_files(spotter, args.wavs)
    if args.mic:
        spotter.reset()
        run_microphone(spotter)


if __name__ == "__main__":
    main()
