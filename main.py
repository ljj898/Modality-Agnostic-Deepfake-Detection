"""
Inference entry point for AVSR-based deepfake detection.

Supports three modes selected automatically by file extension or via --mode:
  - AO  (audio-only):  .wav / .mp3
  - VO  (video-only):  --mode VO with a video file
  - AV  (audio+video): .mp4 / .mov / .avi (default for video inputs)
"""

import argparse
import json
import os
import time
from datetime import datetime

import numpy as np
import torch

from config import args
from data.utils import collate_fn, prepare_main_input
from models.av_net_DualLabel import AVNet_Global_Dropout_ModalComps
from models.visual_frontend import VisualFrontend
from utils.preprocessing import preprocess_sample


AUDIO_EXTS = (".wav", ".mp3")
VIDEO_EXTS = (".mp4", ".mov", ".avi")
SUPPORTED_EXTS = AUDIO_EXTS + VIDEO_EXTS


def parse_args():
    parser = argparse.ArgumentParser(
        description="Modality-Agnostic Deepfake Detection (audio-only, video-only, audio-visual)."
    )
    parser.add_argument("--input_path", type=str, required=True,
                        help="Path to an input media file, or a folder when --d is set.")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Directory where result.json will be saved.")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to the trained detection model checkpoint (.pt).")
    parser.add_argument("--frontend_path", type=str, default=args["TRAINED_FRONTEND_FILE"],
                        help="Path to the visual frontend weights (.pt).")
    parser.add_argument("--info_path", type=str, default="method_info.json",
                        help="Path to method_info.json describing this analytic.")
    parser.add_argument("--mode", type=str, default=None, choices=["AO", "VO", "AV"],
                        help="Force a specific inference mode. Defaults to AO for audio files, AV for video files.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help='"cuda" or "cpu".')
    parser.add_argument("--d", action="store_true",
                        help="Set this flag when --input_path points to a folder.")
    return parser.parse_args()


def get_result_description(real_prob):
    if real_prob >= 0.99:
        return "This sample is certainly real."
    if real_prob >= 0.75:
        return "This sample is likely real."
    if real_prob >= 0.25:
        return "This sample is maybe real."
    if real_prob >= 0.01:
        return "This sample is unlikely real."
    return "There is no chance that the sample is real."


def load_models(checkpoint_path, frontend_path, device):
    model = AVNet_Global_Dropout_ModalComps(
        args["TX_NUM_FEATURES"], args["TX_ATTENTION_HEADS"], args["TX_NUM_LAYERS"],
        args["PE_MAX_LENGTH"], args["AUDIO_FEATURE_SIZE"], args["TX_FEEDFORWARD_DIM"],
        args["TX_DROPOUT"], args["NUM_CLASSES"], args["TOKEN_DROPOUT"], args["ADD_MODAL"],
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device).eval()

    vf = VisualFrontend()
    vf.load_state_dict(torch.load(frontend_path, map_location=device))
    vf.to(device).eval()
    return model, vf


def infer_mode(input_path, forced_mode):
    if forced_mode is not None:
        return forced_mode
    return "AO" if input_path.lower().endswith(AUDIO_EXTS) else "AV"


def run_inference(input_path, output_path, info, model, vf, mode, device):
    start_time = time.time()
    filename = os.path.basename(input_path)
    print(f"Processing {input_path} (mode={mode})")

    temp_dir = "./temp"
    os.makedirs(temp_dir, exist_ok=True)

    params = {
        "roiSize": args["ROI_SIZE"],
        "normMean": args["NORMALIZATION_MEAN"],
        "normStd": args["NORMALIZATION_STD"],
        "vf": vf,
    }
    preprocess_sample(input_path, params)

    stem = os.path.splitext(filename)[0]
    audio_file = os.path.join(temp_dir, stem + "_16.wav")
    visual_features_file = os.path.join(temp_dir, stem + ".npy")
    if mode == "AO":
        visual_features_file = None

    audio_params = {
        "stftWindow": args["STFT_WINDOW"],
        "stftWinLen": args["STFT_WIN_LENGTH"],
        "stftOverlap": args["STFT_OVERLAP"],
    }
    video_params = {"videoFPS": args["VIDEO_FPS"]}
    inp, inp_len = prepare_main_input(
        audio_file, visual_features_file, args["MAIN_REQ_INPUT_LENGTH"],
        audio_params, video_params,
    )

    dummy_label = torch.tensor([0])
    input_batch, input_len_batch, _ = collate_fn([((inp[0], inp[1]), inp_len, dummy_label)])
    input_batch = (input_batch[0].float().to(device), input_batch[1].float().to(device))
    input_len_batch = input_len_batch.int().to(device)

    if mode == "AO":
        input_batch = (input_batch[0], None)
    elif mode == "VO":
        input_batch = (None, input_batch[1])

    with torch.no_grad():
        output_batch, _ = model(input_batch, input_len_batch)

    predicted_a = torch.sigmoid(output_batch[:, 1])
    predicted_v = torch.sigmoid(output_batch[:, 0])
    predicted_b = 1 - ((1 - predicted_a) * (1 - predicted_v))

    print("Fake probabilities -> Audio: %.4f, Visual: %.4f, Video-level: %.4f"
          % (predicted_a, predicted_v, predicted_b))

    if mode == "AO":
        fake_prob_tensor = predicted_a
    elif mode == "VO":
        fake_prob_tensor = predicted_v
    else:  # AV
        fake_prob_tensor = predicted_b
    fake_prob = round(float(fake_prob_tensor), 3)
    real_prob = round(1 - fake_prob, 3)

    result = {
        "Task": info["task"],
        "Input File": filename,
        "Analytic Name": info["analytic_name"],
        "Analysis Date": str(datetime.now()),
        "Mode": mode,
        "Result": {"Real Probability": real_prob, "Fake Probability": fake_prob},
        "Result Description": get_result_description(real_prob),
        "Analytic Description": info["analytic_description"],
        "Analysis Scope": info["analysis_scope"],
        "Reference": info["paper_reference"],
        "Code Link": info["code_reference"],
        "Analysis Time in Second": round(time.time() - start_time, 2),
        "Device": device,
    }

    os.makedirs(output_path, exist_ok=True)
    result_path = os.path.join(output_path, "result.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=4)
    print(f"Results saved to {result_path}")

    for fname in os.listdir(temp_dir):
        fpath = os.path.join(temp_dir, fname)
        if os.path.isfile(fpath):
            os.remove(fpath)


def main():
    cli = parse_args()

    np.random.seed(args["SEED"])
    torch.manual_seed(args["SEED"])
    device = cli.device

    with open(cli.info_path, "r") as f:
        info = json.load(f)["AVSRDD"]

    model, vf = load_models(cli.checkpoint_path, cli.frontend_path, device)

    if cli.d:
        for fname in sorted(os.listdir(cli.input_path)):
            if fname.lower().endswith(SUPPORTED_EXTS):
                fpath = os.path.join(cli.input_path, fname)
                run_inference(fpath, cli.output_path, info, model, vf,
                              infer_mode(fpath, cli.mode), device)
    else:
        run_inference(cli.input_path, cli.output_path, info, model, vf,
                      infer_mode(cli.input_path, cli.mode), device)


if __name__ == "__main__":
    main()
