from sklearn.model_selection import train_test_split
import pandas as pd
from keras_preprocessing.image import ImageDataGenerator

IMAGES_DIR = "../airbus-ship-detection/train_v2"
MASKS_DIR = "../airbus-ship-detection/train_masks"
TEST_IMAGES_DIR = "../airbus-ship-detection/test_v2"
CSV_FILE = "../airbus-ship-detection/train_ship_segmentations_v2.csv"
TEST_CSV_FILE = "../airbus-ship-detection/sample_submission_v2.csv"

BATCH_SIZE = 64

drop_row_list = []


def get_train_data():
    # TODO: put df, validation, seed split to parameters

    df = pd.read_csv(CSV_FILE)
    validation_split = 0.2
    seed = 42

    datagen = ImageDataGenerator(
        rescale=1. / 255,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )

    train_df, val_df = train_test_split(df, test_size=validation_split, random_state=seed)

    train_generator = datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=IMAGES_DIR,
        x_col="ImageId",
        y_col=None,
        target_size=(128, 128),
        class_mode=None,
        batch_size=BATCH_SIZE,
        seed=seed
    )

    mask_generator = datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=MASKS_DIR,
        x_col="ImageId",
        y_col=None,
        target_size=(128, 128),
        class_mode=None,
        batch_size=BATCH_SIZE,
        seed=seed
    )

    validation_generator = datagen.flow_from_dataframe(
        dataframe=val_df,
        directory=IMAGES_DIR,
        x_col="ImageId",
        y_col=None,
        target_size=(128, 128),
        class_mode=None,
        batch_size=BATCH_SIZE,
        seed=seed
    )

    validation_mask_generator = datagen.flow_from_dataframe(
        dataframe=val_df,
        directory=MASKS_DIR,
        x_col="ImageId",
        y_col=None,
        target_size=(128, 128),
        class_mode=None,
        batch_size=BATCH_SIZE,
        seed=seed
    )

    return zip(train_generator, mask_generator), zip(validation_generator, validation_mask_generator)


def get_test_data():
    datagen = ImageDataGenerator(rescale=1. / 255)

    test_df = pd.read_csv(TEST_CSV_FILE)

    test_generator = datagen.flow_from_dataframe(
        dataframe=test_df,
        directory=TEST_IMAGES_DIR,
        x_col="ImageId",
        y_col=None,
        target_size=(128, 128),
        class_mode=None,
        batch_size=BATCH_SIZE,
        seed=73
    )

    return test_generator
