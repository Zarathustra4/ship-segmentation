import config as conf
import keras
from processing.metrics import dice_score, dice_loss
from data_preparation.mask_creator import save_all_masks
from inference.test import plot_masks
from processing.trainer import train_unet
from processing.unet import unet
import pandas as pd


def load_model(
    model_path=conf.MODEL_PATH,
    weight_path=conf.TRAINED_WEIGHTS_PATH
):
    model: keras.models.Model = keras.models.load_model(
        model_path,
        custom_objects={"dice_score": dice_score, "dice_loss": dice_loss}
    )
    model.load_weights(weight_path)

    return model


def main(
        create_new_masks=False,
        new_model=False,
        train_model=False,
        test=True
):
    if create_new_masks:
        df = pd.read_csv(conf.CSV_FILE)
        save_all_masks(df)

    if new_model:
        model = unet()
    else:
        model = load_model()

    if train_model:
        train_unet(model, epochs=20)

    if test:
        model = load_model()
        plot_masks(model)


if __name__ == "__main__":
    main(new_model=False, train_model=False)
