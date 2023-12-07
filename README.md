# Ship Semantic Segmentation

## Data Preparation

### Mask creation

In csv file we have 2 field: ImageId, EncodedPixels.
We need to create masks using these
encoded pixels to provide proper segmentation.

Firstly a mask is a flat numpy array. Specified values are set to 255, other are 0.

```
for i in range(0, len(pixels), 2):
    start_pixel = pixels[i] - 1
    n_pixels = pixels[i + 1]
    end_pixel = start_pixel + n_pixels
    mask[start_pixel: end_pixel] = 255
```

After that the mask is reshaped:

``` 
mask.reshape(shape).T
```

Before create and train a model we need to store that masks at a
folder. To do that I have created ```save_all_masks``` function.

The process is time-consuming because of large amount of data.
Perhaps, it is performed only once.

### Data augmentation

To augment data I used horizontal and vertical flips. Also image data is normalized.
``` 
datagen = ImageDataGenerator(
    rescale=1. / 255,
    horizontal_flip=True,
    vertical_flip=True,
)
```

### Validation split

Data is split by sklearn method ```train_test_split```. To do so, we put prepared pandas dataframe.

## Model architecture

U-Net architecture is used to complete this task. The model consists of two parts:

- encoder
- decoder

The encoder is created using pretrained tensorflow layers. To use it you have to install
it ```pip install git+https://github.com/tensorflow/examples.git```

Here the piece of code:

```
base_model = tf.keras.applications.MobileNetV2(input_shape=[128, 128, 3], include_top=False)

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
```

The decoder is a set of ```pix2pix``` layers, which scales back the image:

``` 
[
    pix2pix.upsample(512, 3),  # 4x4 -> 8x8
    pix2pix.upsample(256, 3),  # 8x8 -> 16x16
    pix2pix.upsample(128, 3),  # 16x16 -> 32x32
    pix2pix.upsample(64, 3),  # 32x32 -> 64x64
    pix2pix.upsample(32, 3),  # 64x64 -> 128x128
]
```

```create_unet``` function combines the encoders and decoders with linear and skip connections.
The last layer is a convolutional with 1 filter and sigmoid activation function. It's determine pixel
to be or not to be a part of a ship.

To measure model performance we need to set proper metrics. Accuracy will be a bad idea
because of unbalanced images - only few percent of an image is a ship if an image contains it at all.

In this model I use __dice coefficient__. 
``` 
intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
return K.mean((2. * intersection + smooth) / (union + smooth), axis=0)
```

``` 
model.compile(optimizer=Adam(learning_rate=0.01),
              loss=dice_loss,
              metrics=[dice_coef])
```

The metric is within 0 and 1. The closer it is to 1, the better model performs.

To optimize model parameter I use dice score loss, which is simply:
```1 - dice_coef```

## Performance

## Testing
To test model work we use images from ```test_v2``` directory. I plot model made prediction next to original image.
In such way we may say is model work satisfying or not.
