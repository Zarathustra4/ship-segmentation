# Configuration variables and global constant variables

ORIG_SHAPE = (768, 768)

IMAGES_DIR = "E:\\Programming\\Image Segmentation\\airbus-ship-detection\\train_v2"
MASKS_DIR = "E:\\Programming\\Image Segmentation\\airbus-ship-detection\\train_masks"
TEST_IMAGES_DIR = "E:\\Programming\\Image Segmentation\\airbus-ship-detection\\test_v2"
CSV_FILE = "E:\\Programming\\Image Segmentation\\airbus-ship-detection\\train_ship_segmentations_v2.csv"
TEST_CSV_FILE = "E:\\Programming\\Image Segmentation\\airbus-ship-detection\\sample_submission_v2.csv"

BATCH_SIZE = 32
VALIDATION_PART = 0.2

MODEL_PATH = "E:\\Programming\\Image Segmentation\\ship_segmentation\\model\\unet-model.h5"
TRAINED_WEIGHTS_PATH = "E:\\Programming\\Image Segmentation\\ship_segmentation\\model\\unet-weights.h5"
