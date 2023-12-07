import keras
import numpy as np
from keras.utils import array_to_img
from scipy import ndimage
import pandas as pd

from data_generator import get_test_data
from dice_score import dice_coef
import matplotlib.pyplot as plt
from config import TEST_CSV_FILE


def create_mask(prediction: np.ndarray):
    threshold = 3e-19
    f = np.vectorize(lambda t: 1 if t <= threshold else 0)
    return f(prediction)


def plot_masks(model: keras.models.Model):
    test_data = get_test_data(pd.read_csv(TEST_CSV_FILE), seed=563)

    images = next(test_data)

    predictions = model.predict(images)

    for i in range(0, 32, 4):
        image = images[i]

        mask = create_mask(predictions[i])

        plt.subplot(1, 2, 1)
        plt.imshow(image)

        plt.subplot(1, 2, 2)
        plt.imshow(array_to_img(mask))

        plt.show()


if __name__ == "__main__":
    model: keras.models.Model = keras.models.load_model(
        "model/unet-model.h5",
        custom_objects={"dice_coef": dice_coef}
    )
    model.load_weights("model/unet-weights.h5")

    plot_masks(model)

