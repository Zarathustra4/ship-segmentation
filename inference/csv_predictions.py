import pandas as pd 
import config as conf
import tensorflow as tf 
import keras
from processing.data_generator import get_test_data
import numpy as np
import cv2

from processing.metrics import dice_score, dice_loss


def create_csv_prediction(model: tf.keras.Model):
    df = pd.read_csv(conf.TEST_CSV_FILE)

    test_data_generator = get_test_data(df)

    row_idx = 0
    for images in test_data_generator:
        if row_idx >= len(df):
            break
        predictions = model.predict(images)
        for prediction in predictions:
            encoded_pixels = encode_prediction(prediction)
            df.loc[row_idx, "EncodedPixels"] = encoded_pixels
            row_idx += 1

    df.to_csv(conf.TEST_CSV_FILE, index=False)


def encode_prediction(prediction: np.ndarray):
    prediction = prediction.reshape((128, 128)).T

    prediction = cv2.resize(prediction, (768, 768), interpolation=cv2.INTER_LINEAR)

    prediction = prediction.reshape(768 * 768)

    encoded_pixels = []

    i = 0
    while i < len(prediction):
        if prediction[i] > 0.5:
            encoded_pixels.append(str(i))
            i += 1
            count = 0
            while i < len(prediction) and prediction[i] > 0.5:
                count += 1
                i += 1
            encoded_pixels.append(str(count))
        i += 1

    return ' '.join(encoded_pixels)


if __name__ == "__main__":
    model: keras.models.Model = keras.models.load_model(
        conf.MODEL_PATH,
        custom_objects={"dice_score": dice_score, "dice_loss": dice_loss}
    )
    model.load_weights(conf.TRAINED_WEIGHTS_PATH)
    create_csv_prediction(model)
