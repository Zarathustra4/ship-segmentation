import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix
from config import TARGET_SIZE as target


def encoder():
    """
    Returns a stack of encoding layers.
    The layers are pretrained.
    Use pip install git+https://github.com/tensorflow/examples.git
    :return: encoder model
    """
    base_model = tf.keras.applications.MobileNetV2(input_shape=[target, target, 3], include_top=False)

    layer_names = [
        'block_1_expand_relu',  # 128x128 -> 64x64
        'block_3_expand_relu',  # 64x64 -> 32x32
        'block_6_expand_relu',  # 32x32 -> 16x16
        'block_13_expand_relu',  # 16x16 -> 8x8
        'block_16_project',  # 8x8 -> 4x4
    ]
    base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

    down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)

    down_stack.trainable = False

    return down_stack


def decoder():
    """
    Returns an array of decoding layers
    :return: list of decoding layers
    """
    return [
        pix2pix.upsample(512, 3),  # 4x4 -> 8x8
        pix2pix.upsample(256, 3),  # 8x8 -> 16x16
        pix2pix.upsample(128, 3),  # 16x16 -> 32x32
        pix2pix.upsample(64, 3),  # 32x32 -> 64x64
        pix2pix.upsample(32, 3),  # 64x64 -> 128x128
    ]


def create_unet(down_stack, up_stack):
    """
    Combines encoder and decoder and returns final model
    :param down_stack: encoder model
    :param up_stack: list of decoding layers
    :return: unet model
    """
    inputs = tf.keras.layers.Input(shape=[target, target, 3])

    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])

    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

    # Final layer uses sigmoid activation for logistic separation
    x = tf.keras.layers.UpSampling2D()(x)

    x = tf.keras.layers.Conv2D(filters=1, kernel_size=1, activation="sigmoid")(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def unet():
    """
    :return: Complete unet model
    """
    return create_unet(
        encoder(),
        decoder()
    )


if __name__ == "__main__":
    model = unet()
    model.summary()
