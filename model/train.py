import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Reshape,\
    Conv2D, Conv2DTranspose, BatchNormalization, LeakyReLU, UpSampling2D, MaxPooling2D
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

INPUTS_DIR = "../preprocess/inputs"
IMAGE_DIM = (150,100,3) # 3 color channels
LEARNING_RATE = 1e-3
EPOCHS = 5

def load_images(directory, batch_size=64):
    data = tf.keras.utils.image_dataset_from_directory(
        directory, labels=None, batch_size=batch_size, image_size=(IMAGE_DIM[0],IMAGE_DIM[1]),
        seed=42, validation_split=0.7, subset='training' # change this to the full dataset later
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
            y_hat, mu, logvar = model(x)
            loss = loss_function(y_hat, y, mu, logvar)
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
                samples.append(x.numpy()/255.0)
                samples.append(model(np.expand_dims(x, axis=0))[0][0].numpy())
                samples.append(y.numpy()/255.0)
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
        from_logits=False, reduction=tf.keras.losses.Reduction.SUM
    )
    reconstruction_loss = bce_fn(x, x_hat)
    return reconstruction_loss

def loss_function(x_hat, x, mu, logvar):
    """
    Computes the negative variational lower bound loss term of the VAE (refer to formulation in notebook).
    Returned loss is the average loss per sample in the current batch.

    Inputs:
    - x_hat: Reconstructed input data of shape (N, H, W, C)
    - x: Input data for this timestep of shape (N, H, W, C)
    
    Returns:
    - loss: Tensor containing the scalar loss for the negative variational lowerbound
    """
    #loss = tf.math.reduce_mean(bce_function(x, x_hat) + dkl_function(mu, logvar))
    loss = tf.math.reduce_mean(bce_function(x, x_hat))
    #loss = tf.keras.losses.MeanSquaredError()(x, x_hat)
    return loss

class Downsample(tf.keras.models.Sequential):
    def __init__(self, filters, batchnorm=True):
        super().__init__()
        self.add(Conv2D(filters, 3, activation='relu', padding='same'))
        self.add(MaxPooling2D((2, 2), padding='same'))
        if batchnorm:
            self.add(BatchNormalization())
class Upsample(tf.keras.models.Sequential):
    def __init__(self, filters):
        super().__init__()
        self.add(Conv2DTranspose(filters, 3, activation='tanh', padding='same'))
        self.add(UpSampling2D((2, 2)))
class VAE(tf.keras.Model):
    def __init__(self, image_shape, latent_size=64, hidden_dim=512):
        super(VAE, self).__init__()
        self.image_shape = image_shape
        self.input_size = image_shape[0]*image_shape[1]*image_shape[2]  # H*W*C
        self.latent_size = latent_size  # Z
        self.hidden_dim = hidden_dim  # H_d
        self.mu_layer = Dense(latent_size) # not used right now
        self.logvar_layer = Dense(latent_size) # not used right now
        self.latent_layer = Dense(latent_size) # not used right now

        self.enc1 = Downsample(64)
        self.enc2 = Downsample(32)
        self.enc3 = Downsample(16, False)

        self.dec1 = Upsample(64)
        self.dec2 = Upsample(32)
        self.dec3 = Upsample(16)


    def call(self, x):
        """
        Performs forward pass through autoencoder model
    
        Inputs:
        - x: Batch of input images of shape (N, C, H, W)
        
        Returns:
        - x_hat: Reconstruced input data of shape (N,C,H,W)
        - mu: Matrix representing estimated posterior mu (N, Z), with Z latent space dimension
        - logvar: Matrix representing estimataed variance in log-space (N, Z), with Z latent space dimension
        """
        
        d1 = self.enc1(x)
        d2 = self.enc2(d1)
        d3 = self.enc3(d2)

        u1 = self.dec1(d3)
        u2 = self.dec2(u1)
        u3 = self.dec3(u2)
        x_hat = Conv2D(3, 3, activation='sigmoid', padding='valid')(u3)
        x_hat = tf.keras.layers.Cropping2D((0, 1))(x_hat)

        return x_hat, 0, 0

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
    model = VAE(IMAGE_DIM, 15)
    print("Loading data")
    x_data = load_images(INPUTS_DIR)
    y_data = load_images(INPUTS_DIR) # later change to be different directory
    print("Training model...")
    for e in range(EPOCHS):
        loss = train_vae(model, x_data, y_data)
        print("epoch loss:",loss)
    print("Saving sample outputs")
    show_vae_images(model, x_data, y_data)
    print("Saving model")
    model.save_weights("./ckpts/model_ckpts")