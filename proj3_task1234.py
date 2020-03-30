"""
COSC 525 - Deep Learning
Project #3: Building Networks with Tensorflow and Keras
Contributors: Anna-Maria Nau and Christoph Metzner
Date: 03/10/20
"""

##########################################
# Import libraries
##########################################
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import sys

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from keras.utils import to_categorical
from keras.datasets import fashion_mnist


##########################################
# Import data
##########################################
# Import fashion-mnist dataset from https://keras.io/datasets/#fashion-mnist-database-of-fashion-articles
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
# Change format of label dataset --> from 1 to [0,1,0,0,0,0,0,0,0,0]
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)


# from https://www.tensorflow.org/tutorials/keras/classification
# creat list with all respective class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']



############################################################
# Data Preprocessing: Min-max Normalization, Reshape
############################################################
# Get min, and max value from training data set using all 60,000 samples
max_val = x_train.max()
min_val = x_train.min()


# Normalization function
def normalization(matrix, max_val, min_val):
    new_matrix = np.array([((image - min_val) / (max_val - min_val)) for image in matrix])
    return new_matrix


# Normalize training (x_train) and testing (x_test) data based on maximal/minimal values of the training data
x_train_scaled = normalization(x_train, max_val, min_val)
x_test_scaled = normalization(x_test, max_val, min_val)


# Reshape data
image_size = x_train.shape[1]
reshaped_x_train_scaled = np.reshape(x_train, [-1, image_size, image_size, 1])
reshaped_x_test_scaled = np.reshape(x_test, [-1, image_size, image_size, 1])



############################################################
# Classes and Functions
############################################################
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


def run_model(model, x_train, y_train, x_test, y_test_cat, y_test, class_names):
    # compile the given model architecture
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())

    # call class for taking time stamps for each epoch
    time_callback_train = TimeHistory()

    # fit the model (train the network) and save metrics in variable history
    history = model.fit(x_train, y_train, epochs=50, batch_size=200, callbacks=[time_callback_train])

    # store time stamps per epoch in variable
    times_train = time_callback_train.times
    print()
    print("Reported times per epoch: \n ", times_train)

    accumulated_time = acc_time(times_train)

    # Evaluate model using testing dataset
    test_start_time = time.time()
    test_loss, test_acc = model.evaluate(x_test, y_test_cat, batch_size=200, verbose=2)
    print()
    print()
    print('Test Loss: {} and Test Accuracy: {}'.format(test_loss, test_acc))
    test_end_time = time.time() - test_start_time
    print('Time for Testing Data: ', test_end_time)

    # Generate prediction output from model using scaled testing dataset
    # Using keras predict_classes
    predictions = model.predict_classes(x_test, verbose=2)

    # Creating the confusion matrix using sklearn library
    conf_matrix = confusion_matrix(y_test, predictions)
    # Changing confusing matrix output using pandas library
    conf_matrix = pd.DataFrame(conf_matrix)
    # Changing index / columns names to class names of output
    conf_matrix.columns = class_names
    conf_matrix.index = class_names

    print()
    print()
    print('10-Class Confusion Matrix:')
    print(conf_matrix)

    # show plots
    # Plot training accuracy values vs epochs
    plt.plot(history.history['accuracy'])
    plt.title('Model Accuracy vs. Epochs')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('acc_epoch.png')
    plt.show()

    # Plot training loss values vs epochs
    plt.plot(history.history['loss'])
    plt.title('Model Loss vs. Epochs')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('loss_epoch.png')
    plt.show()

    # Plot training time vs epochs
    plt.plot(accumulated_time, history.history['loss'])
    plt.title('Model Loss vs. Time')
    plt.ylabel('Loss')
    plt.xlabel('Time in seconds')
    plt.legend(['Train'], loc='upper left')
    plt.savefig('time_epoch.png')
    plt.show()

    return history


#######################################################
# Task 1: Fully Connected Neural Network
#######################################################
FC_model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28), name="FC_Input_Layer"),
    keras.layers.Dense(784, activation='tanh', name="FC_Hidden_Layer_1"),
    keras.layers.Dense(512, activation='sigmoid', name="FC_Hidden_Layer_2"),
    keras.layers.Dense(100, activation='relu', name="FC_Hidden_Layer_3"),
    keras.layers.Dense(10, activation='softmax', name="FC_Output_Layer")],
    name='Fully_Connected_NN')



#######################################################
# Task 2: Small Convolutional Neural Network
#######################################################
Small_CNN = Sequential()
Small_CNN.add(layers.Conv2D(filters=40, kernel_size=(5, 5), strides=1, padding='valid', activation='relu',
                            input_shape=(28, 28, 1)))
Small_CNN.add(layers.MaxPooling2D((2, 2)))
Small_CNN.add(layers.Flatten())
Small_CNN.add(layers.Dense(100, activation='relu'))
Small_CNN.add(layers.Dense(10, activation='softmax'))



#######################################################
# Task 3: Bigger Convolutional Neural Network
#######################################################
bigger_CNN = Sequential(name='Bigger_CNN')
bigger_CNN.add(layers.Conv2D(filters=48, kernel_size=(3, 3), strides=1, padding='valid', activation='relu',
                             input_shape=(28, 28, 1)))
bigger_CNN.add(layers.MaxPooling2D((2, 2,)))
bigger_CNN.add(layers.Conv2D(filters=96, kernel_size=(3, 3), strides=1, padding='valid', activation='relu'))
bigger_CNN.add(layers.MaxPooling2D((2, 2,)))
bigger_CNN.add(layers.Flatten())
bigger_CNN.add(layers.Dense(100, activation='relu'))
bigger_CNN.add(layers.Dense(10, activation='softmax'))



#######################################################
# Task 4: Your Own Convolutional Neural Network
#######################################################
# Built the CNN architecture
own_CNN = Sequential(name='Own_CNN')
own_CNN.add(layers.Conv2D(filters=30, kernel_size=(3, 3), strides=1, padding='valid', activation='relu',
                          input_shape=(28, 28, 1)))
own_CNN.add(layers.MaxPooling2D((2, 2,)))
own_CNN.add(layers.Dropout(rate=0.2))
own_CNN.add(layers.Flatten())
own_CNN.add(layers.Dense(100, activation='relu'))
own_CNN.add(layers.Dense(10, activation='softmax'))


# Driver code main()
def main(argv=None):
    if argv[1] == 'task1':
        # Task 1: Fully Connected Network
        run_model(FC_model, x_train_scaled, y_train_cat, x_test_scaled, y_test_cat, y_test, class_names)

    elif argv[1] == 'task2':
        # Task 2: Small CNN
        run_model(Small_CNN, reshaped_x_train_scaled, y_train_cat, reshaped_x_test_scaled, y_test_cat, y_test, class_names)

    elif argv[1] == 'task3':
        # Task 3: Bigger CNN
        run_model(bigger_CNN, reshaped_x_train_scaled, y_train_cat, reshaped_x_test_scaled, y_test_cat, y_test, class_names)

    elif argv[1] == "task4":
        # Task 4: Own CNN
        run_model(own_CNN, reshaped_x_train_scaled, y_train_cat, reshaped_x_test_scaled, y_test_cat, y_test, class_names)


if __name__ == '__main__':
    main(sys.argv)