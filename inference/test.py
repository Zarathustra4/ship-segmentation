import keras
import numpy as np
from keras.utils import array_to_img
import pandas as pd

from processing.data_generator import get_test_data
from processing.dice_score import dice_coef
import matplotlib.pyplot as plt
from config import TEST_CSV_FILE


def create_mask(prediction: np.ndarray):
    threshold = 7e-20
    f = np.vectorize(lambda t: 1 if t <= threshold else 0)
    return f(prediction)


def plot_masks(model: keras.models.Model):
    test_data = get_test_data(pd.read_csv(TEST_CSV_FILE), seed=345)

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
    import config as conf

    model: keras.models.Model = keras.models.load_model(
        conf.MODEL_PATH,
        custom_objects={"dice_coef": dice_coef}
    )
    model.load_weights(conf.TRAINED_WEIGHTS_PATH)

    plot_masks(model)

