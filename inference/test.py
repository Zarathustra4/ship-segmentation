import keras
import numpy as np
import pandas as pd

from data_preparation.data_generator import get_test_data
import matplotlib.pyplot as plt
import config as conf
from inference.ui_wrapper import get_trained_model


def create_mask(prediction):
    f = np.vectorize(lambda x: 255 if x > 0.5 else 0)
    return f(prediction)


def plot_masks(model: keras.models.Model, n_images=20):
    """
    Plots some sample of predicted data
    :param model: keras trained_model
    :param n_images: int - number of images
    :return: None
    """
    test_data = get_test_data(pd.read_csv(conf.TEST_CSV_FILE))

    count = 0
    for images in test_data:
        if count >= n_images:
            break

        predictions = model.predict(images)

        # Go through predictions and plot their predictions
        for i in range(0, conf.BATCH_SIZE, 1):
            if count >= n_images:
                break

            image = images[i]

            plt.subplot(1, 3, 1)
            plt.imshow(image)

            plt.subplot(1, 3, 2)
            plt.imshow(predictions[i])

            plt.subplot(1, 3, 3)
            plt.imshow(create_mask(predictions[i]))

            plt.show()

            count += 1


if __name__ == '__main__':
    model = get_trained_model()
    plot_masks(model, 20)
