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

def val_model(model, val_x, val_true):
    np.random.seed(42)
    acc_list = []
    for file in os.listdir(val_x):
        lowercased = file.lower()
        if(lowercased.endswith((".jpg", ".jpeg", ".png"))):
            pic = np.array(Image.open(val_x + "/" + file)) / 255.0
            actual = np.array(Image.open(val_true + "/" + file)) / 255.0
            # print(actual.shape)
            # print(pic.shape)
            model_out = model(np.expand_dims(pic, axis=0))[0].numpy()
            if OUTPUT_IS_BW:
                model_out = np.tile(model_out, (1,1,3))

            # print(model_out.shape)
            # print(model_out)
            model_colorless = np.transpose(model_out, (2, 0, 1))[0]
            # print(f"Unsquizen dimension: {model_out.shape}")
            # print(f"squizen {model_colorless.shape}")

            actual_colorspace = np.expand_dims(actual, axis=-1) * np.ones((1, 1, 3), dtype=int)

            samples = [pic, model_out, actual_colorspace]

            flattened_out = model_colorless.flatten()
            # print(flattened_out.shape)
            flattened_true = actual.flatten()
            # print(flattened_true.shape)
            # flattened_true = [1 for _ in flattened_out]
            ##### Uncomment for default grayscale
            # flattened_out = [0.5 for _ in flattened_out]
            ##### Uncomment for random sample
            # flattened_out = np.random.random_sample([len(flattened_out)])
            #####
            # print(f"old length: {len(flattened_out)}")
            # flattened_out = np.random.normal(0.5, 0.5, len(flattened_out))
            # print(f"length: {len(flattened_out)}")
            # print(f"max {np.max(flattened_out)}")
            # print(f"min {np.min(flattened_out)}")

            squared_errors = [(o - t)**2 for o, t in zip(flattened_out, flattened_true)]

            per_pixel_accuracies = [(1-e)**10 for e in squared_errors]
            
            acc_of_img = sum(per_pixel_accuracies)/len(per_pixel_accuracies)

            acc_list.append(acc_of_img)
            print(f"Accuracy of ({file}): {acc_of_img}")

            # Visualize
            fig = plt.figure(figsize=(3, 1))
            gspec = gridspec.GridSpec(1, 3)
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
            plt.savefig(val_x + "/res/" + file, bbox_inches="tight")
            plt.close()
            # plt.show()
    final_acc = sum(acc_list)/len(acc_list)
    print(f"Final Accuracy: {final_acc}")
            
def resize_dir(dir):
    for file in os.listdir(dir):
        lowercased = file.lower()
        if(lowercased.endswith((".jpg", ".jpeg", ".png"))):
            f_img = dir+"/"+file
            img = Image.open(f_img)
            img = img.resize((128,192))
            img.save(f_img)

if __name__ == '__main__':
    model = Autoencoder(IMAGE_DIM)
    model.build(input_shape = (None, 192, 128, 3))
    model.load_weights('./ckpts/model_ckpts')
    # show_image(model, INPUT)
    val_model(model, "../preprocess/val_x", "../preprocess/val_true")
    # resize_dir("../preprocess/val_x")
