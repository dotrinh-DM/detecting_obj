import tensorflow as tf
import numpy as np

def preprocess_image(image_path, img_size=(32, 32)):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=img_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # (1, height, width, channels)
    img_array /= 255.0  # Normalize
    return img_array
