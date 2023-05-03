import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape, Input, Conv2D, MaxPooling2D, UpSampling2D, Reshape, Conv2DTranspose, LeakyReLU, Dense, Flatten
from tensorflow.math import exp, sqrt, square

class VAE(tf.keras.Model):
    def __init__(self, image_shape, latent_size=512):
        super(VAE, self).__init__()
        self.image_shape = image_shape
        self.input_size = image_shape[0]*image_shape[1]*image_shape[2]  # H*W*C
        print(image_shape)
        self.latent_size = latent_size  # Z

        # input_shape = (150, 100, 3)

        self.encoder = tf.keras.Sequential([
            Conv2D(32, (3, 3), padding='same', input_shape=(192, 128, 3)),
            LeakyReLU(alpha=0.2),
            MaxPooling2D((2, 2), padding='same'),
            Conv2D(64, (3, 3), padding='same'),
            LeakyReLU(alpha=0.2),
            MaxPooling2D((2, 2), padding='same'),
            Conv2D(128, (3, 3), padding='same'),
            LeakyReLU(alpha=0.2),
            MaxPooling2D((2, 2), padding='same'),
            Conv2D(256, (3, 3), padding='same'),
            LeakyReLU(alpha=0.2),
            MaxPooling2D((2, 2), padding='same'),
            Flatten(),
            Dense(512),
            LeakyReLU(alpha=0.2)
        ])

        self.flatten = Flatten()
        self.latent = Dense(latent_size, name='latent')
        self.encode_img = Dense(19*13) # from output shap

        self.decoder = tf.keras.Sequential([
            Dense(64 * 24 * 16),
            LeakyReLU(alpha=0.2),
            Reshape((24, 16, 64)),
            Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same'),
            LeakyReLU(alpha=0.2),
            Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same'),
            LeakyReLU(alpha=0.2),
            Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same'),
            # LeakyReLU(alpha=0.2),
            # Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same'),
            LeakyReLU(alpha=0.2),
            Conv2DTranspose(3, (3, 3), padding='same', activation='sigmoid')
        ])

        self.gen_img = Conv2DTranspose(3, 3, activation='sigmoid', padding='same')
        self.crop = tf.keras.layers.Cropping2D((1,2))

    def call(self, x):
        """
        Performs forward pass through autoencoder model
    
        Inputs:
        - x: Batch of input images of shape (N, H, W, C)
        
        Returns:
        - x_hat: Reconstructed input data of shape (N, H, W, C)
        """
        # return x
        # return tf.clip_by_value(Conv2D(3, 1, padding='same')(x), 0, 1)

        # print(x.shape)
    
        # d1 = self.enc1(x)
        # d2 = self.enc2(d1)
        # d3 = self.enc3(d2)

        # l = self.flatten(d3)
        # l = self.latent(l)
        # l = self.encode_img(l)
        # l = self.shape_img(l)

        # u1 = self.dec1(l)
        # u2 = self.dec2(u1)
        # u3 = self.dec3(u2)
        # u3 = self.crop(u3)
        # u3 = tf.keras.layers.Concatenate()([u3, x]) # skip connection
        # x_hat = self.gen_img(u3)

        # print(x.shape)

        x = self.encoder(x)

        # print(x.shape)

        x_hat = self.decoder(x)

        # print(x_hat.shape)

        return x_hat