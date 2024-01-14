from sklearn.model_selection import train_test_split
import pandas as pd
from keras_preprocessing.image import ImageDataGenerator
from config import IMAGES_DIR, BATCH_SIZE, MASKS_DIR, TEST_IMAGES_DIR
from config import TARGET_SIZE as TARGET
import tensorflow as tf


def get_train_data(df: pd.DataFrame,
                   validation_split: float,
                   seed: int = 42):
    """
    Creates data generators of images and masks for model training and validation
    :param df: dataframe with ImageIds
    :param validation_split: percent of validation data
    :param seed: random seed
    :return: train_generator, validation_generator
    """

    if validation_split > 1 or validation_split < 0:
        raise ValueError("validation_split parameter must be within 0 and 1")

    # Use data augmentation and normalization
    datagen = ImageDataGenerator(
        rescale=1. / 255,
        horizontal_flip=True,
        vertical_flip=True,
    )

    train_df, val_df = train_test_split(df, test_size=validation_split)

    train_generator = datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=IMAGES_DIR,
        x_col="ImageId",
        y_col=None,
        target_size=(TARGET, TARGET),
        class_mode=None,
        batch_size=BATCH_SIZE,
        seed=seed
    )

    mask_generator = datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=MASKS_DIR,
        x_col="ImageId",
        y_col=None,
        target_size=(TARGET, TARGET),
        class_mode=None,
        batch_size=BATCH_SIZE,
        seed=seed
    )

    validation_generator = datagen.flow_from_dataframe(
        dataframe=val_df,
        directory=IMAGES_DIR,
        x_col="ImageId",
        y_col=None,
        target_size=(TARGET, TARGET),
        class_mode=None,
        batch_size=BATCH_SIZE,
        seed=seed
    )

    validation_mask_generator = datagen.flow_from_dataframe(
        dataframe=val_df,
        directory=MASKS_DIR,
        x_col="ImageId",
        y_col=None,
        target_size=(TARGET, TARGET),
        class_mode=None,
        batch_size=BATCH_SIZE,
        seed=seed
    )

    return zip(train_generator, mask_generator), zip(validation_generator, validation_mask_generator)


def get_test_data(df: pd.DataFrame, seed: int = 42):
    """
    Returns test image generator
    :param df: pandas dataframe with ImageIds
    :param seed: random seed
    :return: test_generator
    """

    datagen = ImageDataGenerator(rescale=1. / 255)

    test_generator = datagen.flow_from_dataframe(
        dataframe=df,
        directory=TEST_IMAGES_DIR,
        x_col="ImageId",
        y_col=None,
        target_size=(TARGET, TARGET),
        class_mode=None,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return test_generator
