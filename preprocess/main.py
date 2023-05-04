import os
import sys
import tensorflow as tf
os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'
import tensorflow_hub as hub
#import matplotlib.pyplot as plt
#import matplotlib as mpl
#mpl.rcParams['figure.figsize'] = (12, 12)
#mpl.rcParams['axes.grid'] = False
import numpy as np
import PIL.Image

INPUTS = 'photos'
STYLES = 'styles'
ONLY_CHALK = False

# load the model
print("Loading model")
model = hub.load('style-model')

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

def save_img(img, output_path):
  img.save(output_path)

def save_style_transfer(content_path, style_path, output_path):
  content = load_img(content_path)
  style = load_img(style_path)
  stylized = model(tf.constant(content), tf.constant(style))[0]
  stylized = tensor_to_image(stylized)
  save_img(stylized, output_path)

print("Preparing loop")
inputs = os.listdir(INPUTS)
try:
  inputs.remove('.DS_Store')
except: pass
inputs.sort()

styles = os.listdir(STYLES)
try:
  styles.remove('.DS_Store')
except: pass
if ONLY_CHALK:
  styles = ['chalk.jpg']


PRINT_ON = int(len(inputs)*0.05) # print every 5%
if PRINT_ON == 0: PRINT_ON = 1
for i in range(len(inputs)):
  inpt = inputs[i]
  style = np.random.choice(styles)
  #fname = 'paintings/'+inpt[:inpt.rfind('.')]+inpt[inpt.rfind('.')+1:]+'__'+style[:style.rfind('.')]+'.jpg'
  fname = 'paintings/'+inpt
  save_style_transfer(INPUTS+'/'+inpt, STYLES+'/'+style, fname)
  if i % PRINT_ON == 0:
    sys.stdout.write('\rStyling and saving images -- ')
    sys.stdout.write(str(int((i+1)/len(inputs)*100))+"%")
    sys.stdout.flush()
sys.stdout.write('\n')