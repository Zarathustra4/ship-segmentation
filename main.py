import keras
import numpy as np
from keras.utils import array_to_img

from data_generator import get_test_data
from dice_score import dice_coef
import matplotlib.pyplot as plt


def create_mask(prediction: np.ndarray):
    threshold = 8e-19
    f = np.vectorize(lambda t: 1 if t <= threshold else 0)
    return f(prediction)


if __name__ == "__main__":
    model: keras.models.Model = keras.models.load_model(
        "model/unet-model.h5",
        custom_objects={"dice_coef": dice_coef}
    )
    model.load_weights("model/unet-weights.h5")

    test_data = get_test_data()

    next(test_data)
    images = next(test_data)

    predictions = model.predict(images)

    for i in range(0, 64, 4):
        image = images[i]

        mask = create_mask(predictions[i])

        debug_pred = predictions[i].reshape((128, 128))
        min = predictions[i].min(axis=0)
        debug_mask = mask.reshape((128, 128))

        plt.subplot(1, 2, 1)
        plt.imshow(image)

        plt.subplot(1, 2, 2)
        plt.imshow(array_to_img(mask))

        plt.show()
        ...
