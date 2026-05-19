"""
Per-sample preprocessing: extracts 16 kHz mono audio, crops a centered
grayscale lip ROI, and saves visual features through the pretrained
VisualFrontend.

Skeleton adapted from `deep_avsr` (Smeet Shah, 2020).
"""

import os

import cv2 as cv
import numpy as np
import torch


def preprocess_sample(file, params):
    filename = os.path.basename(file)
    stem = os.path.splitext(filename)[0]

    temp_dir = "./temp"
    os.makedirs(temp_dir, exist_ok=True)
    audio_file = os.path.join(temp_dir, stem + "_16.wav")
    roi_file = os.path.join(temp_dir, stem + ".png")
    visual_features_file = os.path.join(temp_dir, stem + ".npy")

    roi_size = params["roiSize"]
    norm_mean = params["normMean"]
    norm_std = params["normStd"]
    vf = params["vf"]
    device = next(vf.parameters()).device

    # Extract 16 kHz mono audio. Quoting the path so spaces don't break ffmpeg.
    os.system(f'ffmpeg -y -v quiet -i "{file}" -ac 1 -ar 16000 -vn "{audio_file}"')

    if file.lower().endswith((".wav", ".mp3")):
        return

    capture = cv.VideoCapture(file)
    roi_sequence = []
    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) / 255.0
        gray = cv.resize(gray, (224, 224))
        roi = gray[
            int(112 - roi_size / 2):int(112 + roi_size / 2),
            int(112 - roi_size / 2):int(112 + roi_size / 2),
        ]
        roi_sequence.append(roi)
    capture.release()

    if not roi_sequence:
        raise RuntimeError(f"No frames decoded from {file}")

    cv.imwrite(roi_file, np.floor(255 * np.concatenate(roi_sequence, axis=1)).astype(int))

    inp = np.stack(roi_sequence, axis=0)
    inp = np.expand_dims(inp, axis=[1, 2])
    inp = (inp - norm_mean) / norm_std
    input_batch = torch.from_numpy(inp).float().to(device)

    vf.eval()
    with torch.no_grad():
        output_batch = vf(input_batch)
    features = torch.squeeze(output_batch, dim=1).cpu().numpy()
    np.save(visual_features_file, features)
