"""Central experiment configuration for Part A."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"
LOGS_DIR = PROJECT_ROOT / "logs"

DATASET = "CIFAR-10"
MODEL_NAME = "resnet18"
NUM_CLASSES = 10

SEED = 42
BATCH_SIZE = 128
NUM_WORKERS = 0
EPOCHS = 50
DEBUG_SUBSET_SIZE = None

LR = 0.1
LR_SCHEDULER = "cosine"
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

SAM_RHO = 0.05

OOD_CORRUPTIONS = ("gaussian_noise", "blur", "brightness")

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

NOISE_STD = 0.15
BLUR_KERNEL_SIZE = 3
BRIGHTNESS_FACTOR = 1.5
