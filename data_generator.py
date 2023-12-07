from sklearn.model_selection import train_test_split
import pandas as pd
from keras_preprocessing.image import ImageDataGenerator
from config import IMAGES_DIR, BATCH_SIZE, MASKS_DIR, TEST_IMAGES_DIR


def get_train_data(df: pd.DataFrame,
                   validation_split: float,
                   seed: int = 42):
    if validation_split > 1 or validation_split < 0:
        raise ValueError("validation_split parameter must be within 0 and 1")

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


def get_test_data(df: pd.DataFrame, seed: int = 42):
    datagen = ImageDataGenerator(rescale=1. / 255)

    test_generator = datagen.flow_from_dataframe(
        dataframe=df,
        directory=TEST_IMAGES_DIR,
        x_col="ImageId",
        y_col=None,
        target_size=(128, 128),
        class_mode=None,
        batch_size=BATCH_SIZE,
        seed=seed
    )

    return test_generator
