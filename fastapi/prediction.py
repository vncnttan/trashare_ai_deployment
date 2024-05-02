from PIL import Image
from io import BytesIO
import numpy as np
import tensorflow as tf

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS = 3

classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash', 'biological']
model = tf.keras.models.load_model('../training/garbage_classification.h5')

def read_image(image):
    image = Image.open(BytesIO(image))
    return image

def preprocess_image(image: Image.Image):
    image = image.resize(IMAGE_SIZE)
    image = np.array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def predict_image(image: np.ndarray):
    prediction = model.predict(image)
    return classes[np.argmax(prediction[0])]