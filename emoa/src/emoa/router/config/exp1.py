from pathlib import Path

# Configuration for the experiment
DATA_DIR = Path("data/processed/routerset_v2")
TRAIN_FILE = DATA_DIR / "train_aug.json"
TEST_FILE = DATA_DIR / "dev.json"
MODEL_SAVE_PATH = Path("ckpt/mf_v2_0.pth")
TRAIN_PLOT_SAVE_PATH = Path("figures/train_mf_v2_0.png")
TEST_PLOT_SAVE_PATH = Path("figures/test_mf_v2_0.png")

PRETRAINED_BERT = "sentence-transformers/all-mpnet-base-v2"  #"microsoft/mdeberta-v3-base"


TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE = 128
EPOCHS = 100
LEARNING_RATE = 1e-5
PATIENCE = 3  # Early stopping patience
MODE = "train"  # Choose "train" or "evaluate" or "predict"

PREDICT_INPUT_FILE = DATA_DIR / "dev.json"
PREDICT_OUTPUT_FILE = DATA_DIR / "dev_pred.json"
NUM_LABELS = 11
