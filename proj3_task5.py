from tensorflow import keras
from keras.layers import Dense, Input
from keras.layers import Conv2D, Flatten, Lambda
from keras.layers import Reshape, Conv2DTranspose
from keras.models import Model
from keras.losses import mse
from keras import backend as K
from keras.datasets import fashion_mnist
from keras.utils import to_categorical

import numpy as np
import matplotlib.pyplot as plt
import os
import time
import sys


# reparameterization trick from https://keras.io/examples/variational_autoencoder/
# instead of sampling from Q(z|X), sample eps = N(0,I)
# then z = z_mean + sqrt(var)*eps
def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.

    # Arguments
        args (tensor): mean and log of variance of Q(z|X)

    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


# normalization function
def normalization(matrix, max_val, min_val):
    new_matrix = np.array([((image - min_val) / (max_val - min_val)) for image in matrix])
    return new_matrix


# from https://stackoverflow.com/questions/43178668/record-the-computation-time-for-each-epoch-in-keras-during-model-fit
# Class which enables to store the execution time per epoch
class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


def acc_time(time_stamps):
    # accumulate all single time values
    accumulated_time = []
    acc_time = 0
    for time in time_stamps:
        acc_time += time
        accumulated_time.append(acc_time)
    # print(accumulated_time)
    return accumulated_time


# function that compiles, fits and plots
def run_vae(model, x_train, x_test, decoder, latent_dim):
    # compile the given model architecture
    model.compile(optimizer='sgd')

    # call class for taking time stamps for each epoch
    time_callback_train = TimeHistory()

    # fit the model (train the network) and save metrics in variable history
    history = model.fit(x_train, epochs=50, batch_size=200, validation_data=(x_test, None),
                        callbacks=[time_callback_train])

    # store time stamps per epoch in variable
    times_train = time_callback_train.times
    print()
    print("Reported times per epoch: \n ", times_train)

    accumulated_time = acc_time(times_train)

    # Evaluate model using testing dataset
    test_start_time = time.time()
    test_loss = model.evaluate(x_test, batch_size=200, verbose=2)
    print()
    print()
    print('Test Loss: {}'.format(test_loss))
    test_end_time = time.time() - test_start_time
    print('Time for Testing Data: ', test_end_time)

    # Plot training loss values vs epochs
    plt.plot(history.history['loss'])
    plt.title('Model Loss vs. Epochs')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('vae_loss_epoch.png')
    plt.show()

    # Plot training time vs loss
    plt.plot(accumulated_time, history.history['loss'])
    plt.title('Model Loss vs. Time')
    plt.ylabel('Loss')
    plt.xlabel('Time in seconds')
    plt.legend(['Train'], loc='upper left')
    plt.savefig('vae_time_epoch.png')
    plt.show()

    # Generate a set of clothes by randomly choosing 10 latent vectors and presenting the resulting images
    random_latents = []
    for i in range(10):
        random_latent = np.random.normal(0, 1, latent_dim)
        decoded_img = decoder.predict(np.array([random_latent]))
        decoded_img_reshape = np.reshape(decoded_img, (28, 28))
        random_latents.append(decoded_img_reshape)

    for i in range(10):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.gray()
        plt.grid(False)
        plt.imshow(random_latents[i])
    plt.savefig('latent_vec_img.png')
    plt.show()

    return history



# import fashion MNIST dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Get min, and max value from training data set using all 60,000 samples
max_val = x_train.max()
# print("Maximal value: ", max_val)
min_val = x_train.min()

# Normalize training (x_train) and testing (x_test) data based on maximal/minimal values of the training data
x_train_scaled = normalization(x_train, max_val, min_val)
x_test_scaled = normalization(x_test, max_val, min_val)

# reshape data
image_size = x_train_scaled.shape[1]
reshaped_x_train_scaled = np.reshape(x_train_scaled, [-1, image_size, image_size, 1])
reshaped_x_test_scaled = np.reshape(x_test_scaled, [-1, image_size, image_size, 1])


#######################################################
# Variational Autoencoder #1
#######################################################
def vae_1(x_train, x_test):
    # network parameters
    input_shape = (image_size, image_size, 1)
    kernel_size = 3
    filters = 16
    latent_dim = 10

    # build encoder
    inputs = Input(shape=input_shape, name='encoder_input')
    x = inputs
    for i in range(2):
        filters *= 2
        x = Conv2D(filters=filters, kernel_size=kernel_size, activation='relu', strides=2, padding='same')(x)

    # shape info needed to build decoder model
    shape = K.int_shape(x)

    # generate latent vector Q(z|X)
    x = Flatten()(x)
    x = Dense(16, activation='relu')(x)
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)

    # use reparameterization trick to push the sampling out as input
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

    # instantiate encoder
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    print(encoder.summary())

    # build decoder
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = Dense(shape[1] * shape[2] * shape[3], activation='relu')(latent_inputs)
    x = Reshape((shape[1], shape[2], shape[3]))(x)

    for i in range(2):
        x = Conv2DTranspose(filters=filters, kernel_size=kernel_size, activation='relu', strides=2, padding='same')(x)
        filters //= 2

    outputs = Conv2DTranspose(filters=1, kernel_size=kernel_size, activation='sigmoid', padding='same', name='decoder_output')(x)

    # instantiate decoder
    decoder = Model(latent_inputs, outputs, name='decoder')
    print(decoder.summary())

    # instantiate VAE
    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name='vae')
    print(vae.summary())

    # set loss
    reconstruction_loss = mse(K.flatten(inputs), K.flatten(outputs))
    reconstruction_loss *= image_size * image_size
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)

    run_vae(vae, x_train, x_test, decoder, latent_dim)



#######################################################
# Variational Autoencoder #2
#######################################################
def vae_2(x_train, x_test):
    # network parameters
    input_shape = (image_size, image_size, 1)
    kernel_size = 3
    filters = 16
    latent_dim = 32

    # VAE model = encoder + decoder
    # build encoder
    inputs = Input(shape=input_shape, name='encoder_input')
    x = inputs
    for i in range(2):
        filters *= 2
        x = Conv2D(filters=filters, kernel_size=kernel_size, activation='relu', strides=2, padding='same')(x)

    # shape info needed to build decoder
    shape = K.int_shape(x)

    # generate latent vector Q(z|X)
    x = Flatten()(x)
    x = Dense(16, activation='relu')(x)
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)

    # use reparameterization trick to push the sampling out as input
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

    # instantiate encoder
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    print(encoder.summary())

    # build decoder
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = Dense(shape[1] * shape[2] * shape[3], activation='relu')(latent_inputs)
    x = Reshape((shape[1], shape[2], shape[3]))(x)

    for i in range(2):
        x = Conv2DTranspose(filters=filters, kernel_size=kernel_size, activation='relu', strides=2, padding='same')(x)
        filters //= 2

    outputs = Conv2DTranspose(filters=1, kernel_size=kernel_size, activation='sigmoid', padding='same', name='decoder_output')(x)

    # instantiate decoder
    decoder = Model(latent_inputs, outputs, name='decoder')
    print(decoder.summary())

    # instantiate VAE
    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name='vae')
    print(vae.summary())

    # set loss
    reconstruction_loss = mse(K.flatten(inputs), K.flatten(outputs))
    reconstruction_loss *= image_size * image_size
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)

    run_vae(vae, x_train, x_test, decoder, latent_dim)

# Driver code main()
def main(argv=None):
    if argv[1] == "task5_vae1":
        # VAE 1
        vae_1(reshaped_x_train_scaled, reshaped_x_test_scaled)

    elif argv[1] == "task5_vae2":
        # VAE 2
        vae_2(reshaped_x_train_scaled, reshaped_x_test_scaled)


if __name__ == '__main__':
    main(sys.argv)


