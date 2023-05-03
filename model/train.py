import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Reshape,\
    Conv2D, Conv2DTranspose, BatchNormalization, LeakyReLU,\
    UpSampling2D, MaxPooling2D, Dropout, Concatenate, Input
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

INPUTS_DIR = "../preprocess/photos"
OUTPUTS_DIR = "../preprocess/photos"
IMAGE_DIM = (192, 128, 3) # 3 color channels
LEARNING_RATE = 5e-3
EPOCHS = 5

def load_images(directory, batch_size=128):
    data = tf.keras.utils.image_dataset_from_directory(
        directory, labels=None, batch_size=batch_size, image_size=(IMAGE_DIM[0],IMAGE_DIM[1]),
        seed=42, validation_split=0.90, subset='training', color_mode='rgb' # change this to the full dataset later
    )
    return data

def train_vae(model, x_data, y_data):
    """
    Train your autoencoder with one epoch.

    Inputs:
    - model: Your VAE instance.
    - x_data: A tf.data.Dataset of noisy input images (stylized images).
    - y_data: A tf.data.Dataset of reconstructed images (photographs).

    Returns:
    - total_loss: Sum of loss values of all batches.
    """
    optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)
    total_loss = 0
    for x, y in zip(x_data, y_data):
        x /= 255.0
        y /= 255.0
        with tf.GradientTape() as tape:
            y_hat = model(x)
            loss = loss_function(y, y_hat)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        total_loss += loss
    return total_loss / len(x_data)

def show_vae_images(model, x_data, y_data):
    """
    Generate 10 images from random vectors.
    Show the generated images from your trained VAE.
    Image will be saved to show_vae_images.pdf

    Inputs:
    - model: Your trained model.
    """
    samples = []
    for xbatch,ybatch in zip(x_data,y_data):
        for x,y in zip(xbatch,ybatch):
            if np.random.random() < 0.1:
                xnorm = x.numpy()/255.0
                ynorm = y.numpy()/255.0
                samples.append(xnorm)
                samples.append(model(np.expand_dims(xnorm, axis=0))[0].numpy())
                samples.append(ynorm)
            if len(samples) >= 9: break
        if len(samples) >= 9: break

    # Visualize
    fig = plt.figure(figsize=(10, 1))
    gspec = gridspec.GridSpec(1, 10)
    gspec.update(wspace=0.05, hspace=0.05)
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
    plt.savefig("show_vae_images.pdf", bbox_inches="tight")
    plt.close(fig)

def mse_function(x, x_hat):
    mse_fn = tf.keras.losses.MeanSquaredError(
        reduction=tf.keras.losses.Reduction.SUM
    )
    reconstruction_loss = mse_fn(x, x_hat)
    return reconstruction_loss

def dkl_function(mu, logvar):
    return -0.5 * tf.math.reduce_sum(1 + logvar - tf.math.square(mu) - tf.math.exp(logvar))

def bce_function(x, x_hat):
    bce_fn = tf.keras.losses.BinaryCrossentropy(
        from_logits=False#, reduction=tf.keras.losses.Reduction.SUM
    )
    # red_loss = bce_fn(x[:,:,0], x_hat[:,:,0])
    # green_loss = bce_fn(x[:,:,1], x_hat[:,:,1])
    # blue_loss = bce_fn(x[:,:,2], x_hat[:,:,2])
    # bright_loss = tf.math.reduce_mean(bce_fn(tf.math.reduce_mean(x, axis=3), tf.math.reduce_mean(x_hat, axis=3)))
    # return (red_loss + green_loss + blue_loss + bright_loss) / 4.0
    return bce_fn(x, x_hat)

def loss_function(x, x_hat):
    """
    Computes the negative variational lower bound loss term of the VAE (refer to formulation in notebook).
    Returned loss is the average loss per sample in the current batch.

    Inputs:
    - x: Input data for this timestep of shape (N, H, W, C)
    - x_hat: Reconstructed input data of shape (N, H, W, C)
    
    Returns:
    - loss: Tensor containing the scalar loss for the negative variational lowerbound
    """
    #loss = tf.math.reduce_mean(bce_function(x, x_hat) + dkl_function(mu, logvar))
    loss = tf.math.reduce_mean(bce_function(x, x_hat))
    #loss = tf.math.reduce_mean(tf.keras.losses.MeanSquaredError()(x, x_hat))
    #loss = tf.math.reduce_mean(tf.random.normal([2]))
    return loss

class Downsample(tf.keras.Sequential):
    def __init__(self, filters, pool=True):
        super().__init__()
        self.add(Conv2D(filters, 3, activation='relu', padding='same'))
        if pool:
            self.add(MaxPooling2D((2, 2), padding='same'))
class Upsample(tf.keras.Sequential):
    def __init__(self, filters, dropout=False):
        super().__init__()
        self.add(Conv2DTranspose(filters, 3, activation='relu', padding='same'))
        self.add(UpSampling2D((2, 2)))
        if dropout:
            self.add(Dropout(0.2))
class VAE(tf.keras.Model):
    def __init__(self, image_shape, latent_size=512):
        super(VAE, self).__init__()
        self.input_size = image_shape[0]*image_shape[1]*image_shape[2]  # H*W*C
        self.latent_size = latent_size  # Z

        self.encoder = Sequential([
            Input(image_shape),
            Downsample(64),
            Downsample(128),
            Downsample(256),
            #Downsample(512),
            #Downsample(256, False),
        ], name='encoder')

        self.latent = Sequential([
            Input(self.encoder.output_shape[1:]),
            Flatten(),
            Dense(latent_size, activation='relu'),
            Dense(24*16*1),
            Reshape((24, 16, 1)), # after three upsamples will be 192x128
        ], name='latent')

        self.decoder = Sequential([
            Input(self.encoder.output_shape[1:]),
            Upsample(256, True),
            Upsample(128),
            Upsample(64)
        ], name='decoder')

        self.skip = Concatenate()

        self.gen_img = Conv2DTranspose(3, 3, activation='sigmoid', padding='same')

    def call(self, x):
        """
        Performs forward pass through autoencoder model
    
        Inputs:
        - x: Batch of input images of shape (N, H, W, C)
        
        Returns:
        - x_hat: Reconstructed input data of shape (N, H, W, C)
        """
    
        enc = self.encoder(x)
        #l = self.latent(enc)
        dec = self.decoder(enc)
        # skip = self.skip([dec, x])
        skip = dec
        x_hat = self.gen_img(skip)

        return x_hat

def reparametrize(mu, logvar):
    """
    Differentiably sample random Gaussian data with specified mean and variance using the
    reparameterization trick.

    Inputs:
    - mu: Tensor of shape (N, Z) giving means
    - logvar: Tensor of shape (N, Z) giving log-variances

    Returns: 
    - z: Estimated latent vectors, where z[i, j] is a random value sampled from a Gaussian with
         mean mu[i, j] and log-variance logvar[i, j].
    """
    eps = tf.random.normal(mu.shape)
    z = mu + tf.math.sqrt(tf.math.exp(logvar))*eps
    return z

if __name__ == "__main__":
    print("Loading model")
    model = VAE(IMAGE_DIM)
    print("Loading data")
    x_data = load_images(INPUTS_DIR)
    y_data = load_images(INPUTS_DIR) # later change to be different directory
    print("Training model...")
    for e in range(EPOCHS):
        loss = train_vae(model, x_data, y_data)
        print("epoch %d/%d loss:" % (e+1, EPOCHS), loss)
        print("Saving sample outputs")
        show_vae_images(model, x_data, y_data)
    print("Saving model")
    model.save_weights("./ckpts/model_ckpts")