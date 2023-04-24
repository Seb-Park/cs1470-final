import os
import sys
import tensorflow as tf
os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'
import tensorflow_hub as hub
import IPython.display as display
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12, 12)
mpl.rcParams['axes.grid'] = False
import numpy as np
import PIL.Image

content_path = 'rembrandt.jpg'
style_path = 'chalk.jpg'

big_img = len(sys.argv) >= 2

def tensor_to_image(tensor):
  tensor = tensor*255
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor)>3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  return PIL.Image.fromarray(tensor)

def load_img(path_to_img, nothing=512):
  max_dim = 512
  max_dim = nothing
  img = tf.io.read_file(path_to_img)
  img = tf.image.decode_image(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)

  shape = tf.cast(tf.shape(img)[:-1], tf.float32)
  long_dim = max(shape)
  scale = max_dim / long_dim

  new_shape = tf.cast(shape * scale, tf.int32)

  img = tf.image.resize(img, new_shape)
  img = img[tf.newaxis, :]
  return img
def imshow(image, title=None):
  if len(image.shape) > 3:
    image = tf.squeeze(image, axis=0)

  plt.imshow(image)
  if title:
    plt.title(title)

img_size = 512
if big_img:
  img_size = 1024
content_image = load_img(content_path, img_size)
style_image = load_img(style_path, 512)

hub_model = hub.load('style-model') # https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2
stylized_image = hub_model(tf.constant(content_image), tf.constant(style_image))[0]

if not big_img:
  plt.figure()
  plt.subplot(1, 3, 1)
  imshow(content_image, 'Content Image')
  plt.axis('off')

  plt.subplot(1, 3, 2)
  imshow(style_image, 'Style Image')
  plt.axis('off')

  tensor_to_image(stylized_image)

  plt.subplot(1, 3, 3)
  imshow(stylized_image, 'Stylized Image')
  plt.axis('off')

  savef = plt.figure()
  tensor_to_image(stylized_image)
  imshow(stylized_image)
  plt.axis('off')
  plt.savefig("./outputs/test.png")
  plt.close(savef)

plt.show()
