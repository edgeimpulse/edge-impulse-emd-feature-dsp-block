import numpy as np
from numpy import genfromtxt
import requests
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer, Dropout, BatchNormalization, TimeDistributed
from tensorflow.keras.optimizers import Adam
import sys, os, random
import tensorflow as tf
from sklearn.model_selection import train_test_split
import logging

dataset = genfromtxt("../dataset/full_dataset_all_stats_features.csv", delimiter=',')
X = dataset[:, 0:80]
Y = dataset[:, 80:81]

print(X)
print(Y)

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

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=250, restore_best_weights=True)
initializer = tf.keras.initializers.GlorotNormal(seed=None)

model = Sequential()
model.add(Dense(200, activation='tanh',
    activity_regularizer=tf.keras.regularizers.l1(0.00001), kernel_initializer=initializer))
model.add(Dropout(0.2))
model.add(Dense(350, activation='tanh',
    activity_regularizer=tf.keras.regularizers.l1(0.00001), kernel_initializer=initializer))
model.add(Dropout(0.2))
model.add(Dense(200, activation='tanh',
    activity_regularizer=tf.keras.regularizers.l1(0.00001), kernel_initializer=initializer))
model.add(Dropout(0.2))
model.add(Dense(12, activation='tanh',
    activity_regularizer=tf.keras.regularizers.l1(0.00001), kernel_initializer=initializer))
model.add(Dense(classes, activation='softmax', name='y_pred'))


opt = Adam(learning_rate=0.0005, beta_1=0.9, beta_2=0.999)

BATCH_SIZE = 32
train_dataset = train_dataset.batch(BATCH_SIZE, drop_remainder=False)
validation_dataset = validation_dataset.batch(BATCH_SIZE, drop_remainder=False)

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
model.fit(train_dataset, epochs=1000000, validation_data=validation_dataset, verbose=2, callbacks=callback)

## Load the model if already saved.
# model = tf.keras.models.load_model('saved_model')

accuracy = model.evaluate(X_test, Y_test, verbose=2);
print((accuracy))

# Use this flag to disable per-channel quantization for a model.
# This can reduce RAM usage for convolutional models, but may have
# an impact on accuracy.
disable_per_channel_quantization = False
model.save('saved_model')
 
