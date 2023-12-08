# Ship Semantic Segmentation

## Data Preparation

### Mask creation

In csv file we have 2 fields: ImageId, EncodedPixels.
We need to create masks using these
encoded pixels to provide proper segmentation.

Firstly, we create a mask as a flat numpy array. Specified values are set to 255, other are 0.

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

![image](https://github.com/Zarathustra4/ship-segmentation/assets/68013193/28908dc7-ab7a-4615-bf52-2aacd29de312)


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

![image](https://github.com/Zarathustra4/ship-segmentation/assets/68013193/688313cd-439f-412b-9100-8a9db00e53ca)


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
The last layer is a convolutional with 1 filter and sigmoid activation function. It determines a pixel
to be or not to be a part of a ship.

To measure model performance we need to set proper metrics. Accuracy is a bad idea
because of unbalanced data - only few percent of an image is a ship if an image contains it at all.

In this model I use __dice coefficient__ as metric. 
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
To train the model we need to run it with such configurations:
```
EPOCHS = 20
STEPS_PER_EPOCH = 200
VALIDATION_STEPS = int(STEPS_PER_EPOCH * VALIDATION_PART)
```

![image](https://github.com/Zarathustra4/ship-segmentation/assets/68013193/614f6cfa-c2c1-4037-b56f-05c8880c77e0)


## Testing
To test model's work we use images from ```test_v2``` directory. I plot model made prediction next to original image.
In such way we may say if model makes satisfying predictions (masks).

![image](https://github.com/Zarathustra4/ship-segmentation/assets/68013193/4a72e561-fddb-4a85-aa4f-052c656c5f17)
![image](https://github.com/Zarathustra4/ship-segmentation/assets/68013193/0215404c-e6e5-4dc4-9c94-c1a0c89a8e75)
![image](https://github.com/Zarathustra4/ship-segmentation/assets/68013193/9df1ec55-5158-44ee-a7f7-bd0bb1ba14c8)
![image](https://github.com/Zarathustra4/ship-segmentation/assets/68013193/e88505d0-199d-46c8-a598-f8ac0d88f66b)
![image](https://github.com/Zarathustra4/ship-segmentation/assets/68013193/4bc11392-3436-4c7b-a9db-97480ea72aa4)
![image](https://github.com/Zarathustra4/ship-segmentation/assets/68013193/93b6bf9d-1179-4b27-883b-fca31e88ed4d)


In ```cvs_predictions``` module you can call ```create_csv_prediction``` method to write down prediction to a csv file in encoded way. 

