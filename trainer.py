import keras
from PIL import ImageFile
from keras.optimizers import Adam
import tensorflow as tf
import pandas as pd

from config import MODEL_PATH, TRAINED_WEIGHTS_PATH, VALIDATION_PART, CSV_FILE
from data_generator import get_train_data
from dice_score import dice_coef
from unet import unet

EPOCHS = 2
STEPS_PER_EPOCH = 100
VALIDATION_STEPS = int(STEPS_PER_EPOCH * VALIDATION_PART)


def train_unet(
        model: keras.models.Model,
        epochs=EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_steps=VALIDATION_STEPS,
        save_model=True
):
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    model.compile(optimizer=Adam(learning_rate=0.01),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=[dice_coef])

    train_generator, validation_generator = get_train_data(pd.read_csv(CSV_FILE),
                                                           validation_split=VALIDATION_PART)

    model.fit(train_generator,
              validation_data=validation_generator,
              epochs=epochs,
              steps_per_epoch=steps_per_epoch,
              validation_steps=validation_steps)

    if save_model:
        model.save(MODEL_PATH)
        model.save_weights(TRAINED_WEIGHTS_PATH)


if __name__ == "__main__":
    model = unet()
    train_unet(model, epochs=1)
