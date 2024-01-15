import keras
import pandas as pd

from config import MODEL_PATH, TRAINED_WEIGHTS_PATH, VALIDATION_PART, CSV_FILE
from inference.ui_wrapper import get_trained_model
from data_preparation.data_generator import get_train_data
from processing.metrics import dice_score, dice_loss
from processing.unet import unet

# Default training values
EPOCHS = 10
STEPS_PER_EPOCH = 1000
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
    :param epochs: number of epochs
    :param model: keras model
    :param steps_per_epoch: number of steps per epoch for training
    :param validation_steps: number of steps per for validation
    :param save_model: if True -> saves a trained model
    :return: history of training
    """
    df = pd.read_csv(CSV_FILE)

    model.compile(optimizer="adam",
                  loss="binary_crossentropy",
                  metrics=[dice_score])

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
    #  If you want to train new model, use:
    # model = unet()

    model = get_trained_model()
    train_unet(model, epochs=10)
