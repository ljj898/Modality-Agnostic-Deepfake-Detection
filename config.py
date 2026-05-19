"""
Configuration for AVSR-based deepfake detection.

The model architecture / preprocessing skeleton was adapted from
`deep_avsr` (https://github.com/lordmartian/deep_avsr) by Smeet Shah.
Detection-specific additions and training data are our own.
"""

args = dict()

# ---- Inference defaults -------------------------------------------------
args["TEST_DEMO_MODE"] = "AV"           # "AO" | "VO" | "AV"; main.py overrides per input
args["ADD_MODAL"] = "A"                  # which modality the dropout-modality branch was trained on

# Paths (relative to the project root). main.py accepts --checkpoint_path / --frontend_path
# CLI flags that override these — keep them in sync if you change the layout.
args["CODE_DIRECTORY"] = "./"
args["DATA_DIRECTORY"] = "./"
args["DEMO_DIRECTORY"] = "./demo"
args["PRETRAINED_MODEL_FILE"] = "./SR_weights/"
args["TRAINED_MODEL_FILE"] = None       # provide via --checkpoint_path
args["TRAINED_FRONTEND_FILE"] = "./SR_weights/visual_frontend.pt"


# ---- Data ---------------------------------------------------------------
args["NUM_WORKERS"] = 4
args["MAIN_REQ_INPUT_LENGTH"] = 145     # minimum input length used at inference
args["CHAR_TO_INDEX"] = {
    " ": 1, "'": 22, "1": 30, "0": 29, "3": 37, "2": 32, "5": 34, "4": 38,
    "7": 36, "6": 35, "9": 31, "8": 33, "A": 5, "C": 17, "B": 20, "E": 2,
    "D": 12, "G": 16, "F": 19, "I": 6, "H": 9, "K": 24, "J": 25, "M": 18,
    "L": 11, "O": 4, "N": 7, "Q": 27, "P": 21, "S": 8, "R": 10, "U": 13,
    "T": 3, "W": 15, "V": 23, "Y": 14, "X": 26, "Z": 28, "<EOS>": 39,
}
args["INDEX_TO_CHAR"] = {v: k for k, v in args["CHAR_TO_INDEX"].items()}


# ---- Audio preprocessing ------------------------------------------------
args["STFT_WINDOW"] = "hamming"
args["STFT_WIN_LENGTH"] = 0.040          # secs
args["STFT_OVERLAP"] = 0.030             # secs


# ---- Video preprocessing ------------------------------------------------
args["VIDEO_FPS"] = 25
args["ROI_SIZE"] = 112
args["NORMALIZATION_MEAN"] = 0.4161
args["NORMALIZATION_STD"] = 0.1688


# ---- Reproducibility ----------------------------------------------------
args["SEED"] = 19220297


# ---- Training (only used by pretrain.py / utils/general.py) -------------
args["PRETRAIN_VAL_SPLIT"] = 0.01
args["PRETRAIN_NUM_WORDS"] = 1
args["BATCH_SIZE"] = 12
args["NUM_STEPS"] = 100
args["SAVE_FREQUENCY"] = 10
args["VIDEO_ONLY_PROBABILITY"] = 0.0
args["INIT_LR"] = 1e-4
args["FINAL_LR"] = 1e-6
args["LR_SCHEDULER_FACTOR"] = 0.5
args["LR_SCHEDULER_WAIT"] = 15


# ---- Model --------------------------------------------------------------
args["AUDIO_FEATURE_SIZE"] = 321
args["NUM_CLASSES"] = 2
args["SR_NUM_CLASSES"] = 40

args["TOKEN_DROPOUT"] = 0.0

# Transformer
args["PE_MAX_LENGTH"] = 2500
args["TX_NUM_FEATURES"] = 512
args["TX_ATTENTION_HEADS"] = 8
args["TX_NUM_LAYERS"] = 6
args["TX_FEEDFORWARD_DIM"] = 2048
args["TX_DROPOUT"] = 0.1


if __name__ == "__main__":
    for key, value in args.items():
        print(f"{key} : {value}")
