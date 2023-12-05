from keras import Input, models
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, UpSampling2D, Concatenate, Dropout, Flatten


def down_block(x, filters, user_max_pool=True):
    x = Conv2D(filters,
               3,
               activation='relu',
               padding='same',
               kernel_initializer='HeNormal')(x)
    x = BatchNormalization()(x)

    if user_max_pool:
        return MaxPooling2D(strides=(2, 2))(x), x

    return x


def up_block(x, y, filters):
    x = UpSampling2D()(x)
    x = Concatenate(axis=3)([x, y])

    x = Conv2D(filters,
               3,
               activation='relu',
               padding='same',
               kernel_initializer='HeNormal')(x)
    x = BatchNormalization()(x)
    return x


def unet(input_size=(256, 256, 1), dropout=0.2):
    filters = [64, 128, 256, 512, 1024]
    input = Input(shape=input_size)
    x, skip1 = down_block(input, filters[0])
    x, skip2 = down_block(x, filters[1])
    x, skip3 = down_block(x, filters[2])
    x, skip4 = down_block(x, filters[3])
    x = down_block(x, filters[4], user_max_pool=False)

    x = up_block(x, skip4, filters[3])
    x = up_block(x, skip3, filters[2])
    x = up_block(x, skip2, filters[1])
    x = up_block(x, skip1, filters[0])

    x = Dropout(dropout)(x)
    output = Conv2D(1, (1, 1), activation='sigmoid')(x)

    model = models.Model(input, output, name='unet')
    return model


if __name__ == "__main__":
    model = unet()
    model.summary()
