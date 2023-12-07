import keras
from keras.optimizers import Adam
import tensorflow as tf
import pandas as pd

from config import MODEL_PATH, TRAINED_WEIGHTS_PATH, VALIDATION_PART, CSV_FILE
from processing.data_generator import get_train_data
from processing.dice_score import dice_coef
from processing.unet import unet

# Default training values
EPOCHS = 10
STEPS_PER_EPOCH = 100
VALIDATION_STEPS = int(STEPS_PER_EPOCH * VALIDATION_PART)


def train_unet(
        model: keras.models.Model,
        epochs=EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_steps=VALIDATION_STEPS,
        save_model=True
):
    """
    Trains model and validates this
    :param model: keras model
    :param epochs: number of epochs
    :param steps_per_epoch: number of steps per epoch for training
    :param validation_steps: number of steps per for validation
    :param save_model: if True -> saves a trained model
    :return: history of training
    """
    df = pd.read_csv(CSV_FILE)

    model.compile(optimizer=Adam(learning_rate=0.01),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=[dice_coef])

    train_generator, validation_generator = get_train_data(df, validation_split=VALIDATION_PART)

    history = model.fit(train_generator,
                        validation_data=validation_generator,
                        epochs=epochs,
                        steps_per_epoch=steps_per_epoch,
                        validation_steps=validation_steps)

    if save_model:
        model.save(MODEL_PATH)
        model.save_weights(TRAINED_WEIGHTS_PATH)

    return history


if __name__ == "__main__":
    model = unet()
    train_unet(model, epochs=1)
