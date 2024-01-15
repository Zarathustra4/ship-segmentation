# Ship Semantic Segmentation

# How to run it:

## Requirements

- python version = 3.9
- `numpy` | `pip install numpy`
- `scikit-learn` | `pip install scikit-learn`
- `pandas` | `pip install pandas`
- `tensorflow` | `pip install "tensorflow<2.11"`
- `keras` | `pip install keras`
- `tensorflow_examples` | `pip install git+https://github.com/tensorflow/examples.git`
- `streamlit` | `pip install streamlit`
- `cv2` | `pip install opencv-python`

### To run the app execute:
```streamlit run main.py```

#### Example:
![image](https://github.com/Zarathustra4/ship-segmentation/assets/68013193/6e01d4fd-775c-4414-8d6e-99cc923b6b2b)

# Solution Approach

## Data Preparation

### Mask creation

In csv file we have 2 fields: ImageId, EncodedPixels.
We need to create masks using these
encoded pixels to provide proper segmentation.

Firstly, we create a mask as a flat numpy array. Specified values are set to 255, other become 0.

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

Before creating and training a model we need to store that masks at a
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

The encoder is created using pretrained tensorflow layers.
Here the piece of code:

```
base_model = tf.keras.applications.MobileNetV2(input_shape=[128, 128, 3], include_top=False)

layer_names = [
    'block_1_expand_relu',  # 256x256 -> 128x128
    'block_3_expand_relu',  # 128x128 -> 64x64
    'block_6_expand_relu',  # 64x64 -> 32x32
    'block_13_expand_relu',  # 32x32 -> 16x16
    'block_16_project',  # 16x16 -> 8x8
]
base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)

down_stack.trainable = False
```

The decoder is a set of ```pix2pix``` layers, which scales back the image:

``` 
[
    pix2pix.upsample(512, 3),  # 8x8 -> 16x16
    pix2pix.upsample(256, 3),  # 16x16 -> 32x32 
    pix2pix.upsample(128, 3),  # 32x32 -> 64x64
    pix2pix.upsample(64, 3),  # 64x64 -> 128x128
    pix2pix.upsample(32, 3),  # 128x128 -> 256x256 
]
```

```create_unet``` function combines the encoders and decoders with linear and skip connections.
The last layer is a convolutional with 1 filter and sigmoid activation function. It determines a pixel
to be or not to be a part of a ship.

To measure model performance we need to set proper metrics. Accuracy is a bad idea
because of unbalanced data - only few percent of an image is a ship if an image contains it at all.

In this model I use __dice coefficient__ as metric. 
```
y_pred = tf.where(y_pred > 0.5, 1.0, 0.0)
intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
return K.mean((2. * intersection + smooth) / (union + smooth), axis=0)
```

``` 
model.compile(optimizer=Adam(learning_rate=0.01),
              loss="binary_crossentropy",
              metrics=[dice_coef])
```

The metric is within 0 and 1. The closer it is to 1, the better model performs.

To optimize model parameter I use binary crossentropy

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
To clear the prediction image I use filter-mask:
``` 
def create_mask(prediction):
    f = np.vectorize(lambda x: 255 if x > 0.5 else 0)
    return f(prediction)
```
![image](https://github.com/Zarathustra4/ship-segmentation/assets/68013193/3f93496f-c5fb-4732-b961-7ba7d077878b)
![image](https://github.com/Zarathustra4/ship-segmentation/assets/68013193/eef44203-713c-4862-863b-5f10814ce4a3)


## The model in ```trained_model``` directory is already trained and ready to use
