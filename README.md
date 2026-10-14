# xiaoguang-kws: 小光 wake word models

Inference code for two small streaming detectors of the Mandarin wake word **小光** (xiǎo guāng). Both also fire on
the repeated form 小光小光. The model weights are on Hugging Face:
[Icerm/xiaoguang-kws](https://huggingface.co/Icerm/xiaoguang-kws).

| Model | What it is | Size | CPU per 80 ms of audio |
|---|---|---|---|
| openWakeWord classifier | A conv-attention head (19k parameters) on openWakeWord's frozen melspectrogram and speech-embedding front end | 88 KB, plus the 2.4 MB shared front end | 1.8 ms |
| sherpa-onnx keyword spotter | k2-fsa's 3.3M-parameter streaming zipformer transducer that outputs pinyin units with tones; its encoder was fine-tuned on 小光 and its near-misses | 13.5 MB (5.4 MB int8) | 1.6 ms |

CPU times are for one thread of a server CPU.

## Quick start

```bash
git clone https://github.com/xiaonengmiao/xiaoguang-kws && cd xiaoguang-kws
uv sync                                  # or: pip install .
uv run xiaoguang-kws recording.wav       # any sample rate; converted to 16 kHz mono

uv sync --extra mic                      # microphone input via sounddevice
uv run xiaoguang-kws --mic
```

The first run downloads the models from Hugging Face, and the openWakeWord front end from the
[openWakeWord v0.5.1 release](https://github.com/dscripka/openWakeWord/releases/tag/v0.5.1) (checksum-verified).
Use `--local-dir` to point at a local copy of the Hugging Face repository instead. Example output:

```
recording.wav
      0.88 s  openWakeWord  小光  (score 0.94)
      1.20 s  sherpa-onnx   小光
```

`--models openwakeword` or `--models sherpa` runs only one model. `xiaoguang-kws --help` lists the thresholds.

## Python API

Both detectors take 16 kHz mono int16 audio in chunks of any size and keep their streaming state between calls.

```python
from xiaoguang_kws import OpenWakeWordDetector, SherpaKwsDetector

oww = OpenWakeWordDetector()   # threshold 0.65
kws = SherpaKwsDetector()      # threshold 0.25, boost 0.2

for chunk in audio_chunks:     # numpy int16 arrays
    if any(score >= oww.threshold for score in oww.process(chunk)):   # one score per completed 80 ms step
        print("小光 (openWakeWord)")
    for keyword in kws.process(chunk):                                 # names of detected keywords
        print(keyword, "(sherpa-onnx)")
```

An openWakeWord score stays above the threshold for several 80 ms steps of one utterance, so add a refractory
period (the command-line tool uses 1 s). The sherpa-onnx spotter reports each keyword once and resets.

## Results

All test audio is synthetic, from speakers and a speech synthesizer that were not used for training:

- **Held-out voices:** VoxCPM2 voice cloning of 20 Emilia reference speakers. About 1,000 clips each of 小光 and 小光小光, and about 250–290 of each near-miss.
- **edge-tts:** Microsoft edge-tts, 11 Mandarin voices × 9 rate and pitch variants, 99 clips per phrase.
- **False activations:** 2 hours of AISHELL-1 test speech that never contains 小光.

Each cell is the share of clips on which a detector fired: the held-out-voice set first, then edge-tts.

| Spoken phrase | openWakeWord (threshold 0.65) | sherpa-onnx (threshold 0.25, boost 0.2) |
|---|---|---|
| 小光 | 95.5 % / 96.0 % | 98.7 % / 87.9 % |
| 小光小光 | 98.7 % / 92.9 % | 99.8 % / 90.9 % |
| 小关 | 7.0 % / 8.1 % | 2.2 % / 0 % |
| 小关小关 | 5.4 % / 4.0 % | 0.4 % / 0 % |
| 小刚, 小黄, 小王 | 0.7 %, 16.8 %, 0.7 % / 1.0 %, 4.0 %, 0 % | 0 %, 1.4 %, 0 % / 0 % |
| 小刚小刚, 小黄小黄, 小王小王 | 1.2 %, 23.9 %, 0.8 % | 0.4 %, 0.9 %, 0.4 % |
| 消光, 小广, 小逛 (tone-only differences) | 80 % / 92 % | 22 % / 11 % |
| False activations per hour of speech | 0.5 | 0 |

- **openWakeWord** accepts accented speech more readily. It hardly separates tones, because its frozen speech embedding encodes Mandarin tone weakly, and it confuses 小黄 more often.
- **sherpa-onnx** rejects near-misses, tone-only ones included, much better. All its edge-tts misses come from the two dialect-accented voices (Shaanxi, Liaoning). All nine standard Mandarin voices are detected.

## Tuning

- **openWakeWord threshold** (`--oww-threshold`, default 0.65): raise it to fire less often.
- **sherpa-onnx threshold and boost** (`--sherpa-threshold`, default 0.25; `--sherpa-boost`, default 0.2): sherpa-onnx fires when a keyword's units are probable on average. A boosted keyword path that survives the search on a near-miss such as 小关, which shares three of four units with 小光, can clear the threshold. Keep the boost low: at the sherpa-onnx default of 1.0, this fine-tuned model reports near-misses as 小光 far more often.
- **Keywords file** (`--keywords`): one keyword per line, written as pinyin units, e.g. `x iǎo g uāng @小光`. The default file ships with the model.
- **int8 model** (`--int8`): uses the int8-quantized sherpa-onnx model.

## Limitations

- Training and test audio is synthetic. Measure on real recordings from your target microphones, distances and speakers before deploying.
- Neither model was evaluated on far-field or noisy real-world audio.
- The sherpa-onnx model rejects some accented pronunciations of 小光. The openWakeWord model accepts most tone-only near-misses.

## License

- **Code:** [Apache-2.0](LICENSE). `xiaoguang_kws/openwakeword_detector.py` ports the streaming front end of [openWakeWord](https://github.com/dscripka/openWakeWord) (Apache-2.0).
- **Model weights on Hugging Face:** [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/), non-commercial. The training data includes CC BY-NC material: the Emilia reference voices and ESC-50. The sherpa-onnx model is a fine-tune of k2-fsa's Apache-2.0 model.
- **openWakeWord front end:** these models are downloaded from the openWakeWord release under that project's terms. Its README lists its pre-trained models as CC BY-NC-SA 4.0.
- **Synthetic voices:** the training speech imitates the voices of real Emilia speakers. Do not use the models or data for impersonation.

## Acknowledgements

[openWakeWord](https://github.com/dscripka/openWakeWord), [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx),
[icefall](https://github.com/k2-fsa/icefall), [VoxCPM](https://github.com/OpenBMB/VoxCPM),
[Emilia](https://huggingface.co/datasets/amphion/Emilia-Dataset), [AISHELL-1](https://www.openslr.org/33/),
[ESC-50](https://github.com/karolpiczak/ESC-50) and [livekit-wakeword](https://github.com/livekit/livekit-wakeword).
