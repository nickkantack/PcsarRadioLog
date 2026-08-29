import torch

SAMPLE_RATE = 16000

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TEN_FOUR_NAME = "segment_1784136495949162_0051bfa0-ed51-440d-9eae-b658645d6f15"
LONG_REPORT_NAME = "segment_1784144726501455_270cd5ca-e1e4-407a-b683-03c76d60a39d"

VALIDATION_SPLIT = 0.2  # 20% for validation, 80% for training
BATCH_SIZE = 2

DENOISER_LEARNING_RATE = 2e-4
EPOCHS_TO_TRAIN = 1000

CHECKPOINT_PREFIX = "small-en-0001lr_"