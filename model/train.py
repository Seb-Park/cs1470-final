import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Reshape,\
    Conv2D, Conv2DTranspose, BatchNormalization, LeakyReLU, ReLU,\
    UpSampling2D, MaxPooling2D, Dropout, Concatenate, Input
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

INPUTS_DIR = "../preprocess/paintings"
INPUT_IS_BW = False
OUTPUTS_DIR = "../preprocess/photos_bw"
OUTPUT_IS_BW = True
IMAGE_DIM = (192, 128, 3) # 3 color channels
LEARNING_RATE = 3e-3
EPOCHS = 10

def load_images(directory, batch_size=128, color_mode='rgb'):
    data = tf.keras.utils.image_dataset_from_directory(
        directory, labels=None, batch_size=batch_size, image_size=(IMAGE_DIM[0],IMAGE_DIM[1]),
        seed=42, validation_split=0.1, subset='training', color_mode=color_mode
    )
    return data

def train_epoch(model, x_data, y_data):
    """
    Train your autoencoder with one epoch.

    Inputs:
    - model: Your autoencoder instance.
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

def show_images(model, x_data, y_data):
    """
    Generate 9 images from random samples.
    Show the generated images from your trained model.
    Image will be saved to show_images.pdf

    Inputs:
    - model: Your trained model.
    """
    samples = []
    for xbatch,ybatch in zip(x_data,y_data):
        for x,y in zip(xbatch,ybatch):
            if np.random.random() < 0.1:
                xnorm = x.numpy()/255.0
                ynorm = y.numpy()/255.0
                if INPUT_IS_BW:
                    samples.append(np.tile(xnorm, (1,1,3)))
                else:
                    samples.append(xnorm)
                model_out = model(np.expand_dims(xnorm, axis=0))[0].numpy()
                if OUTPUT_IS_BW:
                    samples.append(np.tile(model_out, (1,1,3)))
                    samples.append(np.tile(ynorm, (1,1,3)))
                else:
                    samples.append(model_out)
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
    plt.savefig("show_images.pdf", bbox_inches="tight")
    plt.close(fig)

def mse_function(x, x_hat):
    mse_fn = tf.keras.losses.MeanSquaredError()
    reconstruction_loss = mse_fn(x, x_hat)
    return reconstruction_loss

def mae_function(x, x_hat):
    mae_fn = tf.keras.losses.MeanAbsoluteError()
    reconstruction_loss = mae_fn(x, x_hat)
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
    Computes the loss of the autoencoder.
    Returned loss is the average loss per sample in the current batch.

    Inputs:
    - x: Input data for this timestep of shape (N, H, W, C)
    - x_hat: Reconstructed input data of shape (N, H, W, C)
    
    Returns:
    - loss: Tensor containing the scalar loss for the negative variational lowerbound
    """
    variance_bias = 3
    #loss = tf.math.reduce_mean(bce_function(x, x_hat) + dkl_function(mu, logvar))
    #loss = tf.math.reduce_mean(bce_function(x, x_hat))
    loss = tf.math.reduce_mean(bce_function(x, x_hat) + mse_function(x, x_hat) - variance_bias*tf.math.reduce_variance(x_hat))
    #loss = tf.math.reduce_mean(tf.keras.losses.MeanSquaredError()(x, x_hat))
    #loss = tf.math.reduce_mean(tf.random.normal([2]))
    return loss

class Downsample(tf.keras.Sequential):
    def __init__(self, filters, pool=True):
        super().__init__()
        self.add(Conv2D(filters, 3, padding='same'))
        self.add(LeakyReLU())
        if pool:
            self.add(MaxPooling2D((2, 2), padding='same'))
class Upsample(tf.keras.Sequential):
    def __init__(self, filters, unpool=True, dropout=False):
        super().__init__()
        self.add(Conv2DTranspose(filters, 3, padding='same'))
        self.add(BatchNormalization())
        self.add(LeakyReLU())
        if unpool:
            self.add(UpSampling2D((2, 2)))
        if dropout:
            self.add(Dropout(0.2))
class Autoencoder(tf.keras.Model):
    def __init__(self, image_shape, latent_size=256):
        super(Autoencoder, self).__init__()
        self.input_size = image_shape[0]*image_shape[1]*image_shape[2]  # H*W*C
        self.latent_size = latent_size  # Z

        self.enc1 = Downsample(64)
        self.enc2 = Downsample(64)
        self.enc3 = Downsample(128)
        self.enc4 = Downsample(256)
        self.enc5 = Downsample(256)

        # self.latent = Sequential([
        #     Input(self.encoder.output_shape[1:]),
        #     Flatten(),
        #     Dense(latent_size, activation='relu'),
        #     Dense(24*16*3),
        #     Reshape((24, 16, 3)), # after three upsamples will be 192x128
        # ], name='latent')

        self.dec5 = Upsample(256, dropout=True)
        self.dec4 = Upsample(128, dropout=True)
        self.dec3 = Upsample(64)
        self.dec2 = Upsample(64)
        self.dec1 = Upsample(8)

        self.skip = Concatenate()

        self.gen_img = Sequential([
            Conv2D(3, 5, padding='same'),
            Conv2D(1 if OUTPUT_IS_BW else 3, 1, activation='sigmoid', padding='same')
        ])

    def call(self, x):
        """
        Performs forward pass through autoencoder model
    
        Inputs:
        - x: Batch of input images of shape (N, H, W, C)
        
        Returns:
        - x_hat: Reconstructed input data of shape (N, H, W, C)
        """

        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        #e3 = self.enc3(e2)
        #e4 = self.enc4(e3)
        #e5 = self.enc5(e4)

        #d5 = self.dec5(e5)
        #skip5 = self.skip([d5, e4])
        #d4 = self.dec4(skip5)
        #skip4 = self.skip([d4, e3])
        #d3 = self.dec3(e3)
        #skip3 = self.skip([d3, e2])
        d2 = self.dec2(e2)
        skip2 = self.skip([d2, e1])
        d1 = self.dec1(skip2)
        #skip1 = self.skip([d1, x])

        x_hat = self.gen_img(d1)

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
    model = Autoencoder(IMAGE_DIM)
    print("Loading data")
    x_data = load_images(INPUTS_DIR, color_mode = 'grayscale' if INPUT_IS_BW else 'rgb')
    y_data = load_images(OUTPUTS_DIR, color_mode = 'grayscale' if OUTPUT_IS_BW else 'rgb')
    print("Training model...")
    for e in range(EPOCHS):
        loss = train_epoch(model, x_data, y_data)
        print("epoch %d/%d loss:" % (e+1, EPOCHS), loss)
        show_images(model, x_data, y_data)
        print("Saved sample outputs")
    print("Saving model")
    model.save_weights("./ckpts/model_ckpts")
