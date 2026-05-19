# Modality-Agnostic Deepfake Detection (AVSR-based)

Audio-visual deepfake detector with a single model that supports three inference
modes:

- **AO** — audio-only (`.wav`, `.mp3`)
- **VO** — video-only (visual stream of an `.mp4` / `.mov` / `.avi`)
- **AV** — full audio + visual (default for video files)

The model uses dual-branch encoders for audio and video so each modality can be
scored independently. The visual frontend and the architecture skeleton are
adapted from [deep_avsr](https://github.com/lordmartian/deep_avsr); the
deepfake-detection head and training are our own.

Paper: <https://dl.acm.org/doi/10.1145/3733102.3733133>

## Installation

Requires Python 3.8+ and `ffmpeg` on the system `PATH` (used to extract 16 kHz
mono audio from video files).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

A CUDA-capable GPU is recommended; the code falls back to CPU automatically.

## Pretrained weights

Two checkpoints are needed and are **not** stored in the repository:

| File | Purpose | Where to get it |
| --- | --- | --- |
| `SR_weights/visual_frontend.pt` | Pretrained lip-region visual frontend | From the upstream [deep_avsr SR_weights](https://github.com/lordmartian/deep_avsr) |
| Detection checkpoint, e.g. `./best_model.pt` | The deepfake detection model | Released alongside the paper — see the [project page / release section] |

Place them anywhere on disk and point at them with the CLI flags below.

## Usage

Single file:

```bash
python main.py \
  --input_path  xxx.mp4 \
  --output_path ./output_folder \
  --checkpoint_path ./best_model.pt \
  --frontend_path   ./SR_weights/visual_frontend.pt
```

Folder of files (audio and/or video, processed individually):

```bash
python main.py --d \
  --input_path  /path/to/folder \
  --output_path ./output_folder \
  --checkpoint_path ./best_model.pt
```

Force a specific mode (overrides the extension-based default):

```bash
python main.py --mode VO --input_path clip.mp4 --output_path out --checkpoint_path ckpt.pt
```

### Output

A `result.json` is written into `--output_path`:

```json
{
  "Task": "Dual Label Deepfake Video Detection",
  "Input File": "xxx.mp4",
  "Mode": "AV",
  "Result": { "Real Probability": 0.123, "Fake Probability": 0.877 },
  "Result Description": "There is no chance that the sample is real.",
  "Analysis Time in Second": 4.21,
  "Device": "cuda"
}
```

Stdout additionally shows the per-modality fake probabilities:

```
Fake probabilities -> Audio: 0.8459, Visual: 0.9998, Video-level: 1.0000
```

## Input expectations

- Video: 25 fps, frontal face roughly centered in the frame, no occlusion.
- Audio: mono is preferred (otherwise `ffmpeg` will downmix to mono at 16 kHz).
- Length: at least a few seconds. The preprocessor pads short inputs.

## Project layout

```
config.py                 inference / model hyperparameters
main.py                   CLI entry point
method_info.json          analytic metadata embedded in result.json
models/
  av_net_DualLabel.py     dual-label AV detection model
  visual_frontend.py      lip-region visual frontend (from deep_avsr)
  composition_classifier.py
data/
  utils.py                STFT extraction, collate functions
                          edit to match your data layout)
utils/
  preprocessing.py        per-sample audio extraction + ROI + features
  general.py              train / evaluate loops
  decoders.py             CTC greedy / beam-search decoding
  metrics.py              CER / WER
  Dual*.py / LF_*.py      training-loop variants used for ablations
```

## Citation

```bibtex
@inproceedings{cai2025modality,
  title={Modality-agnostic deepfakes detection},
  author={Cai, Yu and Chen, Peng and Tian, Jiahe and Liu, Jin and Dai, Jiao and Wang, Xi and Jia, Shan and Lyu, Siwei and Han, Jizhong},
  booktitle={Proceedings of the 2025 ACM Workshop on Information Hiding and Multimedia Security},
  pages={12--23},
  year={2025}
}
```

## Acknowledgements

- [deep_avsr](https://github.com/lordmartian/deep_avsr) by Smeet Shah for the
  visual frontend and AVSR architecture skeleton.
