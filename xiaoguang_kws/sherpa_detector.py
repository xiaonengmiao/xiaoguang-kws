"""小光 keyword spotting with the fine-tuned sherpa-onnx KWS zipformer (wenetspeech 3.3M)."""
import tempfile
from pathlib import Path
from typing import List

import numpy as np

from .models import SHERPA_SUBDIR, model_dir

SAMPLE_RATE = 16000


def read_keywords(path) -> List[str]:
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.lstrip().startswith("#")]


class SherpaKwsDetector:
    """
    Streaming keyword spotter; `process` returns the names of keywords detected in the audio passed so far.

    Keywords are lines of pinyin units (initials + toned finals) with an optional ':boost', '#threshold' and
    '@name', e.g. "x iǎo g uāng @小光"; the default keywords file ships with the model. Keep the boost low (0.2):
    with the sherpa-onnx default of 1.0 the keyword path survives the search on near-misses such as 小关, and the
    fine-tuned model then reports them as the wake word.
    """

    def __init__(self, threshold: float = 0.25, boost: float = 0.2, keywords_file: str = None,
                 local_dir: str = None, int8: bool = False, num_threads: int = 1):
        import sherpa_onnx
        directory = model_dir(SHERPA_SUBDIR, local_dir)

        def find(prefix):
            files = sorted(p for p in directory.glob(f"{prefix}-*.onnx") if p.name.endswith(".int8.onnx") == int8)
            if not files:
                raise FileNotFoundError(f"no {prefix}-*.onnx in {directory}")
            return str(files[0])

        lines = read_keywords(keywords_file or directory / "keywords.txt")
        self.names = [line.split("@", 1)[1].strip() for line in lines if "@" in line]
        # sherpa-onnx does not skip comment lines, so it gets a copy without them
        with tempfile.NamedTemporaryFile("w", suffix=".txt", encoding="utf-8", delete=False) as f:
            f.write("\n".join(lines) + "\n")
        try:
            self.spotter = sherpa_onnx.KeywordSpotter(
                tokens=str(directory / "tokens.txt"), encoder=find("encoder"), decoder=find("decoder"),
                joiner=find("joiner"), keywords_file=f.name, num_threads=num_threads,
                keywords_threshold=threshold, keywords_score=boost, provider="cpu")
        finally:
            Path(f.name).unlink()
        self.reset()

    def reset(self):
        self._stream = self.spotter.create_stream()

    def process(self, pcm16: np.ndarray) -> List[str]:
        self._stream.accept_waveform(SAMPLE_RATE, np.asarray(pcm16, dtype=np.float32) / 32768)
        detected = []
        while self.spotter.is_ready(self._stream):
            self.spotter.decode_stream(self._stream)
            keyword = self.spotter.get_result(self._stream)
            if keyword:
                detected.append(keyword)
                self.spotter.reset_stream(self._stream)
        return detected
