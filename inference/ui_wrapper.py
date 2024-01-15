import keras
import numpy as np

import config as conf
import streamlit as st
import cv2

from processing.metrics import dice_score, dice_loss
from config import TARGET_SIZE as TARGET
import tensorflow as tf


def create_mask(prediction):
    f = np.vectorize(lambda x: 255 if x > 0.3 else 0)
    return f(prediction)


def get_trained_model():
    model: keras.models.Model = keras.models.load_model(
        conf.MODEL_PATH,
        custom_objects={"dice_score": dice_score, "dice_loss": dice_loss}
    )
    model.load_weights(conf.TRAINED_WEIGHTS_PATH)
    return model


def prepare_image(image):
    resized_cv = tf.image.resize(image, (TARGET, TARGET))
    resized_numpy = np.array(resized_cv)

    return resized_numpy.reshape(1, TARGET, TARGET, 3) / 255.0


def smooth_prediction(prediction):
    small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    mid_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    large_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))

    small_resize_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mid_resize_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    large_resize_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    small_kernel_smoothed = cv2.morphologyEx(prediction, cv2.MORPH_OPEN, small_kernel, iterations=1)
    mid_kernel_smoothed = cv2.morphologyEx(prediction, cv2.MORPH_OPEN, mid_kernel, iterations=1)
    large_kernel_smoothed = cv2.morphologyEx(prediction, cv2.MORPH_OPEN, large_kernel, iterations=1)

    small_smooth = large_kernel_smoothed + mid_kernel_smoothed + small_kernel_smoothed

    resized = cv2.resize(small_smooth, (768, 768), interpolation=cv2.INTER_NEAREST)

    small_resized_smoothed = cv2.morphologyEx(resized, cv2.MORPH_OPEN, small_resize_kernel, iterations=1)
    mid_resized_smoothed = cv2.morphologyEx(resized, cv2.MORPH_OPEN, mid_resize_kernel, iterations=3)
    large_resized_smoothed = cv2.morphologyEx(resized, cv2.MORPH_OPEN, large_resize_kernel, iterations=5)

    return small_resized_smoothed + mid_resized_smoothed + large_resized_smoothed


def prepare_mask(prediction):
    smoothed = smooth_prediction(prediction)
    smoothed = cv2.resize(smoothed, (768, 768), interpolation=cv2.INTER_NEAREST)
    prediction = cv2.resize(prediction, (768, 768), interpolation=cv2.INTER_NEAREST)
    return create_mask(smoothed)


def run_ui():
    model = get_trained_model()

    st.title("Image Semantic Segmentation Model")
    st.divider()

    uploaded_image = st.file_uploader("Upload 768x768 image", type="jpg", )

    if uploaded_image is None:
        return

    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    st.image(image, channels="BGR", width=768, caption="Your image")
    image = prepare_image(image)

    prediction = model.predict(image)[0]
    mask = prepare_mask(prediction)

    st.image(mask, width=768, caption="Smoothed")
