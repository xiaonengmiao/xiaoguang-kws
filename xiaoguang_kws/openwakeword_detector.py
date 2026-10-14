"""小光 / 小光小光 wake word: openWakeWord front end + a trained conv-attention classifier.

The streaming front end is a port of `AudioFeatures` and `Model.predict` from openWakeWord
(https://github.com/dscripka/openWakeWord, Apache-2.0) and produces the same scores. Audio is processed in
80 ms steps: each step adds 8 frames of a 32-bin melspectrogram and one 96-dim speech embedding of the last 76
frames, and the classifier scores the last 16 embeddings (about 1.3 s of audio).
"""
from typing import List

import numpy as np
import onnxruntime as ort

from .models import OPENWAKEWORD_SUBDIR, feature_model, model_dir

SAMPLE_RATE = 16000
STEP = 1280  # 80 ms
CONTEXT = 480  # 3 melspectrogram hops of audio before each step
MEL_WINDOW = 76
N_EMBEDDINGS = 16


class OpenWakeWordDetector:
    """Scores 16 kHz int16 audio; `process` returns one score in [0, 1] per completed 80 ms step."""

    def __init__(self, threshold: float = 0.65, local_dir: str = None, num_threads: int = 1):
        options = ort.SessionOptions()
        options.intra_op_num_threads = num_threads
        options.inter_op_num_threads = 1

        def session(path):
            return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])

        self._melspectrogram = session(feature_model("melspectrogram.onnx", local_dir))
        self._embedding = session(feature_model("embedding_model.onnx", local_dir))
        self._classifier = session(model_dir(OPENWAKEWORD_SUBDIR, local_dir) / "xiaoguang_both.onnx")
        self._classifier_input = self._classifier.get_inputs()[0].name
        self.threshold = threshold
        self.reset()

    def reset(self):
        self._pending = np.zeros(0, dtype=np.int16)
        self._history = np.zeros(0, dtype=np.int16)
        self._mel = np.ones((MEL_WINDOW, 32), dtype=np.float32)
        # As in openWakeWord, the embedding history starts from noise; the first 5 scores are reported as 0
        noise = np.random.default_rng(0).integers(-1000, 1000, SAMPLE_RATE * 4).astype(np.int16)
        mel = self._mel_frames(noise)
        windows = [mel[i:i + MEL_WINDOW] for i in range(0, len(mel) - MEL_WINDOW + 1, 8)]
        self._embeddings = self._embed(np.stack(windows))
        self._steps = 0

    def _mel_frames(self, pcm16: np.ndarray) -> np.ndarray:
        spec = self._melspectrogram.run(None, {"input": pcm16.astype(np.float32)[None]})[0]
        return np.squeeze(spec) / 10 + 2

    def _embed(self, windows: np.ndarray) -> np.ndarray:
        out = self._embedding.run(None, {"input_1": windows[..., None].astype(np.float32)})[0]
        return out.reshape(len(windows), -1)

    def process(self, pcm16: np.ndarray) -> List[float]:
        audio = np.concatenate([self._pending, np.asarray(pcm16, dtype=np.int16)])
        n_steps = len(audio) // STEP
        self._pending = audio[n_steps * STEP:]
        if n_steps == 0:
            return []
        new = audio[:n_steps * STEP]
        self._mel = np.vstack([self._mel, self._mel_frames(np.concatenate([self._history, new]))])[-970:]
        self._history = np.concatenate([self._history, new])[-CONTEXT:]
        windows = [self._mel[len(self._mel) - 8 * i - MEL_WINDOW:len(self._mel) - 8 * i] for i in range(n_steps - 1, -1, -1)]
        self._embeddings = np.vstack([self._embeddings, self._embed(np.stack(windows))])[-120:]
        scores = []
        for i in range(n_steps - 1, -1, -1):
            features = self._embeddings[len(self._embeddings) - N_EMBEDDINGS - i:len(self._embeddings) - i]
            score = float(self._classifier.run(None, {self._classifier_input: features[None]})[0].squeeze())
            scores.append(score if self._steps >= 5 else 0.0)
            self._steps += 1
        return scores
