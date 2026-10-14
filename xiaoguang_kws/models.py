"""Locate or download the model files."""
import hashlib
import os
import urllib.request
from pathlib import Path

HF_REPO = os.environ.get("XIAOGUANG_KWS_REPO", "Icerm/xiaoguang-kws")
OPENWAKEWORD_SUBDIR = "openwakeword"
SHERPA_SUBDIR = "sherpa-onnx-kws-zipformer-wenetspeech-3.3M-xiaoguang-ft"

# Shared openWakeWord front end (melspectrogram + Google speech embedding), downloaded from the openWakeWord release
FEATURE_MODELS = {
    "melspectrogram.onnx": "ba2b0e0f8b7b875369a2c89cb13360ff53bac436f2895cced9f479fa65eb176f",
    "embedding_model.onnx": "70d164290c1d095d1d4ee149bc5e00543250a7316b59f31d056cff7bd3075c1f",
}
FEATURE_MODEL_URL = "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/{}"


def cache_dir() -> Path:
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "xiaoguang_kws"


def model_dir(subdir: str, local_dir: str = None) -> Path:
    """Directory with the files of one model: `local_dir/subdir` if given, else the Hugging Face download."""
    if local_dir:
        path = Path(local_dir) / subdir
        if not path.is_dir():
            raise FileNotFoundError(f"{path} does not exist")
        return path
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(HF_REPO, allow_patterns=[f"{subdir}/*"])) / subdir


def feature_model(name: str, local_dir: str = None) -> Path:
    """Path of an openWakeWord front-end model, downloading and verifying it on first use."""
    for directory in ([Path(local_dir) / OPENWAKEWORD_SUBDIR] if local_dir else []) + [cache_dir() / "openwakeword-v0.5.1"]:
        if (directory / name).is_file():
            return directory / name
    path = cache_dir() / "openwakeword-v0.5.1" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".part")
    urllib.request.urlretrieve(FEATURE_MODEL_URL.format(name), partial)
    digest = hashlib.sha256(partial.read_bytes()).hexdigest()
    if digest != FEATURE_MODELS[name]:
        partial.unlink()
        raise RuntimeError(f"{name}: unexpected checksum {digest}")
    partial.rename(path)
    return path
