import os
import numpy as np
import tensorflow as tf
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import PIL.Image as Image

INPUT = "../preprocess/light_rembdrandt.jpg"
INPUT_IS_BW = False
IMAGE_DIM = (192, 128, 3) # 3 color channels
OUTPUT_IS_BW = True

from train import Autoencoder

def show_image(model, x_dir):
    """
    Generate 9 images from random samples.
    Show the generated images from your trained model.
    Image will be saved to show_images.pdf

    Inputs:
    - model: Your trained model.
    """
    pic = np.array(Image.open(x_dir)) / 255.0
    model_out = model(np.expand_dims(pic, axis=0))[0].numpy()
    if OUTPUT_IS_BW:
        model_out = np.tile(model_out, (1,1,3))
    samples = [pic, model_out]

    # Visualize
    fig = plt.figure(figsize=(2, 1))
    gspec = gridspec.GridSpec(1, 2)
    gspec.update(wspace=0.4, hspace=0.4)
    for i, sample in enumerate(samples):
        ax = plt.subplot(gspec[i])
        plt.axis("off")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect("equal")
        sample *= 255
        sample = sample.astype('int32')
        plt.imshow(sample)

    # Save the generated images
    plt.savefig("show_val_image.pdf", bbox_inches="tight")
    plt.show()

if __name__ == '__main__':
    model = Autoencoder(IMAGE_DIM)
    model.build(input_shape = (None, 192, 128, 3))
    model.load_weights('./ckpts/model_ckpts')
    show_image(model, INPUT)
