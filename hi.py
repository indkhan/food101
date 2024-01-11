import gradio as gr
from tensorflow.keras.models import load_model
import tensorflow as tf


def imageshape(img):

    image_data = tf.io.read_file(img)
    image = tf.image.decode_image(image_data)
    img_shape = 224
    image = tf.image.resize(image, [img_shape, img_shape])

    image = image[:, :, :3]

    image = tf.expand_dims(image, axis=0)
    return image.shape


# Create the interface
interface = gr.Interface(fn=imageshape, inputs="image", outputs="text")

# Launch the interface
interface.launch()
