from os.path import join

import pandas as pd
from keras_preprocessing.image import ImageDataGenerator

IMAGES_DIR = "../airbus-ship-detection/train_v2"
MASKS_DIR = "../airbus-ship-detection/train_masks"
CSV_FILE = "../airbus-ship-detection/train_ship_segmentations_v2.csv"


def get_train_data():
    df = pd.read_csv(CSV_FILE)

    image_datagen = ImageDataGenerator(rescale=1./255)
    mask_datagen = ImageDataGenerator(rescale=1./255)
    seed = 42

    train_generator = image_datagen.flow_from_dataframe(
        dataframe=df,
        directory=IMAGES_DIR,
        x_col="ImageId",  # Column in the DataFrame containing image filenames
        y_col=None,
        target_size=(768, 768),
        class_mode=None,
        batch_size=32,
        seed=seed
    )

    mask_generator = mask_datagen.flow_from_dataframe(
        dataframe=df,
        directory=MASKS_DIR,
        x_col="ImageId",
        y_col=None,
        target_size=(768, 768),
        class_mode=None,
        batch_size=32,
        seed=seed
    )

    return zip(train_generator, mask_generator)
