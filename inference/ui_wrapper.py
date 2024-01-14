import keras
import numpy as np

import config as conf
import streamlit as st
import cv2

from processing.metrics import dice_score, dice_loss
from config import TARGET_SIZE as target
import matplotlib.image


def create_mask(prediction):
    f = np.vectorize(lambda x: 255 if x > 0.5 else 0)
    return f(prediction)


def get_model():
    model: keras.models.Model = keras.models.load_model(
        conf.MODEL_PATH,
        custom_objects={"dice_score": dice_score, "dice_loss": dice_loss}
    )
    model.load_weights(conf.TRAINED_WEIGHTS_PATH)
    return model


def prepare_image(image):
    resized_cv = cv2.resize(image, (target, target))
    resized_numpy = np.array(resized_cv)

    return resized_numpy.reshape(1, target, target, 3) / 255.0


def run_ui():
    model = get_model()

    st.title("Image Semantic Segmentation Model")
    st.divider()

    uploaded_image = st.file_uploader("Upload 768x768 image", type="jpg", )

    if uploaded_image is None:
        return

    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    st.image(image, channels="BGR", width=500, caption="Your image")

    image = prepare_image(image)

    prediction = model.predict(image)

    prediction = create_mask(prediction.reshape(target, target))

    # TODO: Try to save the prediction as an image and see what it is

    matplotlib.image.imsave("prediction.jpg", prediction)

    st.image(prediction, width=500, caption="Mask")
