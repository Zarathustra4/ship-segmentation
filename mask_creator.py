import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from config import ORIG_SHAPE


def generate_image_mask(pixels, shape=ORIG_SHAPE):
    if not pixels:
        return np.zeros(shape)
    if len(pixels) % 2 != 0:
        raise ValueError("Wrong args length, must be even number")

    mask = np.zeros(shape[0] * shape[1])

    for i in range(0, len(pixels), 2):
        start_pixel = pixels[i] - 1
        n_pixels = pixels[i + 1]
        end_pixel = start_pixel + n_pixels
        mask[start_pixel: end_pixel] = 1

    return mask.reshape(shape).T


def generate_mask(pixels, shape=ORIG_SHAPE):
    if not pixels:
        return np.zeros((1, shape[0] * shape[1]))
    if len(pixels) % 2 != 0:
        raise ValueError("Wrong args length, must be even number")

    mask = np.zeros((1, shape[0] * shape[1]))

    for i in range(0, len(pixels), 2):
        start_pixel = pixels[i] - 1
        n_pixels = pixels[i + 1]
        end_pixel = start_pixel + n_pixels
        mask[0, start_pixel: end_pixel] = 1

    return mask


def display_mask(mask: np.ndarray, shape=ORIG_SHAPE):
    mask = mask.reshape(shape).T
    plt.imshow(mask)
    plt.show()


def save_mask(mask: np.ndarray, image_name: str):
    path = f"../airbus-ship-detection/train_masks/{image_name}"
    cv2.imwrite(path, mask)


def save_all_masks(df: pd.DataFrame):
    df['EncodedPixels'] = df['EncodedPixels'].apply(
        lambda pixels: [int(item) for item in pixels.split(" ")] if pd.notna(pixels) else []
    )

    size = len(df)

    count = 0
    for index, row in df.iterrows():
        if count % 100 == 0:
            print(f"[INFO] {count} masks are saved | {count / size * 100: .2f}%")

        save_mask(
            generate_image_mask(row["EncodedPixels"]),
            row["ImageId"]
        )

        count += 1


if __name__ == "__main__":
    import config as conf
    df = pd.read_csv(conf.CSV_FILE, delimiter=",")

    # save_all_masks(df)
