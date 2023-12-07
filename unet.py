import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix

from data_generator import get_train_data

base_model = tf.keras.applications.MobileNetV2(input_shape=[128, 128, 3], include_top=False)

layer_names = [
    'block_1_expand_relu',
    'block_3_expand_relu',
    'block_6_expand_relu',
    'block_13_expand_relu',
    'block_16_project',
]
base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)

down_stack.trainable = False

up_stack = [
    pix2pix.upsample(512, 3),  # 4x4 -> 8x8
    pix2pix.upsample(256, 3),  # 8x8 -> 16x16
    pix2pix.upsample(128, 3),  # 16x16 -> 32x32
    pix2pix.upsample(64, 3),  # 32x32 -> 64x64
    pix2pix.upsample(32, 3),  # 64x64 -> 128x128
]


def unet_model():
    inputs = tf.keras.layers.Input(shape=[128, 128, 3])

    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])

    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

    last = tf.keras.layers.Conv2DTranspose(
        filters=1, kernel_size=3, strides=2,
        activation="sigmoid",
        padding='same')

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


if __name__ == "__main__":
    from dice_score import dice_coef
    from keras.optimizers import Adam
    from PIL import ImageFile
    import keras

    ImageFile.LOAD_TRUNCATED_IMAGES = True

    OUTPUT_CLASSES = 1
    MODEL_PATH = "E:\\Programming\\Image Segmentation\\ship_segmentation\\model\\unet-model.h5"
    TRAINED_WEIGHTS_PATH = "E:\\Programming\\Image Segmentation\\ship_segmentation\\model\\unet-weights.h5"

    model = unet_model()
    model.summary()
    model.compile(optimizer=Adam(learning_rate=0.01),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=[dice_coef])

    train_generator, validation_generator = get_train_data()

    history = model.fit(train_generator,
                        validation_data=validation_generator,
                        epochs=5,
                        steps_per_epoch=100,
                        validation_steps=20)

    model.save(MODEL_PATH)
    model.save_weights(TRAINED_WEIGHTS_PATH)
