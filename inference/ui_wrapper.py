import keras
import numpy as np

import config as conf
import streamlit as st
import cv2

from processing.metrics import dice_score, dice_loss
from config import TARGET_SIZE as TARGET
import tensorflow as tf


def create_mask(prediction):
    """Creates mask of model prediction"""
    f = np.vectorize(lambda x: 255 if x > 0.5 else 0)
    return f(prediction)


def get_trained_model():
    """Loads trained model"""
    model: keras.models.Model = keras.models.load_model(
        conf.MODEL_PATH,
        custom_objects={"dice_score": dice_score, "dice_loss": dice_loss}
    )
    model.load_weights(conf.TRAINED_WEIGHTS_PATH)
    return model


def prepare_image(image):
    """Prepares image for model"""
    resized_cv = tf.image.resize(image, (TARGET, TARGET))
    resized_numpy = np.array(resized_cv)

    return resized_numpy.reshape(1, TARGET, TARGET, 3) / 255.0


def smooth_prediction(prediction):
    """
    Smooths prediction image.
    Use it if model target size is 128x128
    """
    # kernel for small ships at raw prediction
    small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    # kernel for middle ships at raw prediction
    mid_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # kernel for large ships at raw prediction
    large_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))

    # kernel for small ships at resized prediction
    small_resize_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # kernel for middle-sized ships at resized prediction
    mid_resize_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    # kernel for large ships at resized prediction
    large_resize_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    # smoothed predictions at raw prediction
    small_kernel_smoothed = cv2.morphologyEx(prediction, cv2.MORPH_OPEN, small_kernel, iterations=1)
    mid_kernel_smoothed = cv2.morphologyEx(prediction, cv2.MORPH_OPEN, mid_kernel, iterations=1)
    large_kernel_smoothed = cv2.morphologyEx(prediction, cv2.MORPH_OPEN, large_kernel, iterations=1)

    # average value
    small_smooth = (large_kernel_smoothed + mid_kernel_smoothed + small_kernel_smoothed) / 3

    resized = cv2.resize(small_smooth, (768, 768), interpolation=cv2.INTER_NEAREST)

    # smoothed predictions at resized prediction
    small_resized_smoothed = cv2.morphologyEx(resized, cv2.MORPH_OPEN, small_resize_kernel, iterations=1)
    mid_resized_smoothed = cv2.morphologyEx(resized, cv2.MORPH_OPEN, mid_resize_kernel, iterations=3)
    large_resized_smoothed = cv2.morphologyEx(resized, cv2.MORPH_OPEN, large_resize_kernel, iterations=5)

    return (small_resized_smoothed + mid_resized_smoothed + large_resized_smoothed) / 3


def prepare_mask(prediction):
    """Prepares mask. Use it if model target size is 128x128"""
    smoothed = smooth_prediction(prediction)
    smoothed = cv2.resize(smoothed, (768, 768), interpolation=cv2.INTER_NEAREST)
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

    prediction = cv2.resize(prediction, (768, 768))

    mask = create_mask(prediction)
    
    st.image(prediction, caption="Model prediction", width=768)
    st.image(mask, caption="Mask", width=768)
