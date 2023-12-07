import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from config import ORIG_SHAPE, MASKS_DIR
from os.path import join


# The module is created for generating masks from csv file using encoded pixels


def generate_image_mask(pixels, shape=ORIG_SHAPE) -> np.ndarray:
    """
    Generates mask using encoded pixels
    :param pixels: list[int] encoded pixels from csv
    :param shape: tuple[int] shape of image mask, default - 768x768
    :return: np.ndarray - numpy mask
    """
    if not pixels:
        return np.zeros(shape)
    if len(pixels) % 2 != 0:
        raise ValueError("Wrong args length, must be even number")

    mask = np.zeros(shape[0] * shape[1])

    for i in range(0, len(pixels), 2):
        start_pixel = pixels[i] - 1
        n_pixels = pixels[i + 1]
        end_pixel = start_pixel + n_pixels
        mask[start_pixel: end_pixel] = 255

    return mask.reshape(shape).T


def display_mask(mask: np.ndarray, shape=ORIG_SHAPE) -> None:
    """
    Displays mask as image
    :param mask: np.ndarray - numpy mask
    :param shape: shape of the mask, default - 768x768
    :return: None
    """
    mask = mask.reshape(shape).T
    plt.imshow(mask)
    plt.show()


def save_mask(mask: np.ndarray, image_name: str) -> None:
    """
    Saves given mask to directory
    :param mask: numpy mask
    :param image_name: ImageId from csv
    :return: None
    """
    path = join(MASKS_DIR, image_name)
    cv2.imwrite(path, mask)


def save_all_masks(df: pd.DataFrame) -> None:
    """
    Saves all masks to a disc
    :param df: pandas dataframe of csv file
    :return: None
    """
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

    save_all_masks(df)
