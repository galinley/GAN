import os
import time
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt


def preprocess_image(image):
    #low_res_image = tf.image.decode_image(tf.io.read_file(image_path))
    low_res_image = tf.keras.preprocessing.image.img_to_array(image)

    # Todo: Check to see if the low res image is too big and if so,
    # call downscale_image() to make it a bit smaller.

    # If image is a PNG, remove the alpha channel. This model only supports
    # RGB channels.
    if low_res_image.shape[-1] == 4:
        low_res_image = low_res_image[..., :-1]
    img_size = (tf.convert_to_tensor(low_res_image.shape[:-1]) // 4) * 4
    # print("img_size = ", img_size)
    low_res_image = tf.image.crop_to_bounding_box(low_res_image, 0, 0, img_size[0], img_size[1])
    low_res_image = tf.cast(low_res_image, tf.float32)

    return tf.expand_dims(low_res_image, 0)


def save_image(image, filename):
    """
        Args:
          image: 3D image tensor -- [height, width, channels]
          filename: Name of the file to save to
    """
    if not isinstance(image, Image.Image):
        image = tf.clip_by_value(image, 0, 255)
        image = Image.fromarray(tf.cast(image, tf.uint8).numpy())

    image.save("%s.png" % filename, "PNG")
    print("Image saved as %s.png" % filename)

def plot_image(image, title=''):
    dpi = mpl.rcParams['figure.dpi']
    image = np.asarray(image)
    image = tf.clip_by_value(image, 0, 255)
    image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
    width, height = image.size
    figsize = width / float(dpi), height / float(dpi)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])

    # Hide spines, ticks, etc.
    ax.axis('off')
    ax.imshow(image)


def downscale_image(image):
    """
      Scales down images using bicubic downsampling.
      Args:
          image: 3D or 4D tensor of preprocessed image
    """
    image_size = []
    if len(image.shape) == 3:
        image_size = [image.shape[1], image.shape[0]]
    else:
        raise ValueError("Dimension mismatch. Can work only on single image.")

    image = tf.squeeze(
        tf.cast(
            tf.clip_by_value(image, 0, 255), tf.uint8))

    lr_image = np.asarray(
        Image.fromarray(image.numpy())
            .resize([image_size[0] // 4, image_size[1] // 4],
                    Image.BICUBIC))
    print("lr_image shape = ", lr_image.shape)
    lr_image = tf.expand_dims(lr_image, 0)
    lr_image = tf.cast(lr_image, tf.float32)

    return lr_image


def load_model():
    # Load the model from disk (can also use TF-Hub)
    SAVED_MODEL_PATH = "esrgan-tf2_1_SAVEDMODEL"
    model = tf.saved_model.load(SAVED_MODEL_PATH)

    return model


def upscale_image(image, model):
    # Preprocess the low res image
    print("Preprocessing image...")
    lr_image = preprocess_image(image)
    print("Preprocessing complete!")

    start_time = time.time()
    sr_image = model(lr_image)
    sr_image = tf.squeeze(sr_image)
    end_time = (time.time() - start_time)
    print("Time taken to upscale image: %f" % end_time )

    hr_image = np.asarray(sr_image)
    hr_image = tf.clip_by_value(hr_image, 0, 255)
    hr_image = Image.fromarray(tf.cast(hr_image, tf.uint8).numpy())

    return hr_image, end_time
