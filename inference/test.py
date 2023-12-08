import keras
import numpy as np
import pandas as pd

from processing.data_generator import get_test_data
from processing.metrics import dice_score, dice_loss
import matplotlib.pyplot as plt
import config as conf


def create_mask(prediction):
    f = np.vectorize(lambda x: 255 if x > 0.5 else 0)
    return f(prediction)


def plot_masks(model: keras.models.Model):
    """
    Plots some sample of predicted data
    :param model: keras model
    :return: None
    """
    test_data = get_test_data(pd.read_csv(conf.TEST_CSV_FILE), seed=708)

    images = next(test_data)

    predictions = model.predict(images)

    # Go through predictions and plot their predictions
    for i in range(0, conf.BATCH_SIZE, 3):
        image = images[i]

        plt.subplot(1, 3, 1)
        plt.imshow(image)

        plt.subplot(1, 3, 2)
        plt.imshow(predictions[i])

        plt.subplot(1, 3, 3)
        plt.imshow(create_mask(predictions[i]))

        plt.show()


if __name__ == "__main__":
    model: keras.models.Model = keras.models.load_model(
        conf.MODEL_PATH,
        custom_objects={"dice_score": dice_score, "dice_loss": dice_loss}
    )
    model.load_weights(conf.TRAINED_WEIGHTS_PATH)

    plot_masks(model)
