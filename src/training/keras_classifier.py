import numpy as np
from numpy import genfromtxt
import requests
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer, Dropout, Conv1D, Conv2D, Flatten, Reshape, MaxPooling1D, MaxPooling2D, BatchNormalization, TimeDistributed
from tensorflow.keras.optimizers import Adam


API_KEY = 'ei_dbea50a2d038bf91e8cbd06f668e00c1f5596d08576437b07c0a194cef26b294'

def download_data(url):
    response = requests.get(url, headers={'x-api-key': API_KEY})
    if response.status_code == 200:
        return response.content
    else:
        print(response.content)
        raise ConnectionError('Could not download data file')


dataset = genfromtxt("./edge-impulse-emd-feature-dsp-block/src/dataset/full_dataset_all_stats_features.csv", delimiter=',')
X = dataset[:, 0:80]
Y = dataset[:, 80:81]

print(X)
print(Y)

# with open('x_train.npy', 'wb') as file:
#     file.write(X)
# with open('y_train.npy', 'wb') as file:
#     file.write(Y) 
# X = np.load('x_train.npy')
# Y = np.load('y_train.npy')[:,0]

import sys, os, random
import tensorflow as tf
from sklearn.model_selection import train_test_split
import logging
tf.get_logger().setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Set random seeds for repeatable results
RANDOM_SEED = 3
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
classes_values = [ "class1", "class10", "class11", "class2", "class3", "class4", "class5", "class6", "class7", "class8", "class9"]

classes = len(classes_values)
Y = tf.keras.utils.to_categorical(Y - 1, classes)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
input_length = X_train[0].shape[0]
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
validation_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test))

def get_reshape_function(reshape_to):
    def reshape(image, label):
        return tf.reshape(image, reshape_to), label
    return reshape

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=200, restore_best_weights=True)
initializer = tf.keras.initializers.GlorotNormal(seed=None)

# model architecture
model = Sequential()
model.add(Dense(180, activation='tanh',
    activity_regularizer=tf.keras.regularizers.l1(0.00001), kernel_initializer=initializer))
model.add(Dropout(0.2))
model.add(Dense(240, activation='tanh',
    activity_regularizer=tf.keras.regularizers.l1(0.00001), kernel_initializer=initializer))
model.add(Dropout(0.2))
model.add(Dense(180, activation='tanh',
    activity_regularizer=tf.keras.regularizers.l1(0.00001), kernel_initializer=initializer))
model.add(Dropout(0.2))
model.add(Dense(12, activation='tanh',
    activity_regularizer=tf.keras.regularizers.l1(0.00001), kernel_initializer=initializer))
model.add(Dense(classes, activation='softmax', name='y_pred'))

# this controls the learning rate
opt = Adam(learning_rate=0.0005, beta_1=0.9, beta_2=0.999)
# this controls the batch size, or you can manipulate the tf.data.Dataset objects yourself\
BATCH_SIZE = 32
train_dataset = train_dataset.batch(BATCH_SIZE, drop_remainder=False)
validation_dataset = validation_dataset.batch(BATCH_SIZE, drop_remainder=False)

# train the neural network
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
model.fit(train_dataset, epochs=2000, validation_data=validation_dataset, verbose=2, callbacks=callback)

# Use this flag to disable per-channel quantization for a model.
# This can reduce RAM usage for convolutional models, but may have
# an impact on accuracy.

disable_per_channel_quantization = False
model.save('saved_model')
