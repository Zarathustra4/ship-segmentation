import keras
import numpy as np
from keras.utils import array_to_img
import pandas as pd

from processing.data_generator import get_test_data
from processing.dice_score import dice_coef
import matplotlib.pyplot as plt
import config as conf


def create_mask(prediction: np.ndarray):
    """
    Creates mask of given prediction to be displayed
    :param prediction: prediction of a model
    :return: mask of the prediction
    """

    # The threshold is the crucial variable for the mask creation.
    threshold = 5e-18
    f = np.vectorize(lambda t: 1 if t <= threshold else 0)
    return f(prediction)


def plot_masks(model: keras.models.Model):
    """
    Plots some sample of predicted data
    :param model: keras model
    :return: None
    """
    test_data = get_test_data(pd.read_csv(conf.TEST_CSV_FILE), seed=42)

    images = next(test_data)

    predictions = model.predict(images)

    # Go through predictions and plot their masks
    for i in range(0, conf.BATCH_SIZE, 4):
        image = images[i]

        mask = create_mask(predictions[i])

        plt.subplot(1, 2, 1)
        plt.imshow(image)

        plt.subplot(1, 2, 2)
        plt.imshow(array_to_img(mask))

        plt.show()


if __name__ == "__main__":
    model: keras.models.Model = keras.models.load_model(
        conf.MODEL_PATH,
        custom_objects={"dice_coef": dice_coef}
    )
    model.load_weights(conf.TRAINED_WEIGHTS_PATH)

    plot_masks(model)
