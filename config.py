import os

PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = r"E:\Programming\Image Segmentation\airbus-ship-detection"

ORIG_SHAPE = (768, 768)
TARGET_SIZE = 128  # TODO: Try 256


IMAGES_DIR = os.path.join(DATASET_PATH, "train_v2")
MASKS_DIR = os.path.join(DATASET_PATH, "train_masks")
TEST_IMAGES_DIR = os.path.join(DATASET_PATH, "test_v2")
CSV_FILE = os.path.join(DATASET_PATH, "train_ship_segmentations_v2.csv")
TEST_CSV_FILE = os.path.join(DATASET_PATH, "sample_submission_v2.csv")

BATCH_SIZE = 32
VALIDATION_PART = 0.2

MODEL_PATH = os.path.join(PROJECT_PATH, "trained_model", "unet-model.h5")
TRAINED_WEIGHTS_PATH = os.path.join(PROJECT_PATH, "trained_model", "unet-weights.h5")
