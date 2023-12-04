from keras.optimizers import Adam

from unet import unet
from data_generator import get_train_data
from keras.losses import SparseCategoricalCrossentropy


if __name__ == "__main__":
    model = unet()

    train_generator = get_train_data()

    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss=SparseCategoricalCrossentropy(),
                  metrics=['acc'])

    model.fit(train_generator,
              steps_per_epoch=2000,
              epochs=50)
