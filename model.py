import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
import time as time

BATCHSIZE = 16
EPOCHS = 30
log_dir = './logs'


def save_Images():
    data = []
    labels = []

    for folder in os.listdir('./train'):
        for image in os.listdir('./train/' + folder):
            try:
                img = Image.open('./train/' + folder + '/' + image)
                resized_img = img.resize((50, 50))
                if(np.array_equal(np.array(resized_img).shape, [50, 50, 3])):
                    data.append(np.array(resized_img))
                    labels.append(folder)
                else:
                    break
            except AttributeError:
                print('')

    images = np.array(data)
    labels = np.array(labels)
    np.save('images', images)
    np.save('labels', labels)

def trainModel():
    #save_Images()
    images = np.load('images.npy')
    labels = np.load('labels.npy')

    le = LabelEncoder()
    ys = le.fit_transform(labels)

    """with open(os.path.join(log_dir, 'metadata.tsv'), 'w') as f:
        np.savetxt(f, ys)"""

    labels = labels.reshape(-1, 1)
    OneHotEncoder = OneHotEncoder()
    labels = OneHotEncoder.fit_transform(labels)
    labels /= 255

    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size= 0.20, random_state= 3)

    model = Sequential()
    model.add(Conv2D(filters = 64, kernel_size=3, activation= "relu", input_shape= (50,50,3)))
    model.add(Conv2D(filters=64, kernel_size=3, activation= "relu"))
    model.add(MaxPooling2D(pool_size=3))
    model.add(Conv2D(filters= 128, kernel_size= 3, activation= "relu"))
    model.add(Conv2D(filters= 128, kernel_size= 3, activation= "relu"))
    model.add(MaxPooling2D(pool_size= 3))
    model.add(Flatten())
    model.add(Dense(128, activation= "relu", name= 'features'))
    model.add(Dense(12, activation= "softmax"))
    #model.summary()

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    """tensorboard = TensorBoard(batch_size= BATCHSIZE,
                            embeddings_freq= 1,
                            embeddings_layer_names= ['features'],
                            embeddings_metadata= 'metadata.tsv',
                            embeddings_data= X_test)"""
    tensorboard = TensorBoard()

    model.compile(loss= keras.losses.categorical_crossentropy,
                optimizer= keras.optimizers.Adam(),
                metrics= ['accuracy'])

    model.fit(X_train, y_train,
            batch_size= BATCHSIZE,
            callbacks= [tensorboard],
            epochs= EPOCHS,
            verbose= 1,
            validation_data= (X_test, y_test))

    model.save('my_model - {}.h5')

    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    """class TensorBoardWithSession(tf.keras.callbacks.TensorBoard):

        def __init__(self, **kwargs):
            from tensorflow.python.keras import backend as K
            self.sess = K.get_session()

            super().__init__(**kwargs)

        tf.keras.callbacks.TensorBoard = TensorBoardWithSession"""

model = tf.keras.models.load_model('my_model.h5')

#tensorboard --logdir=./logs
#http://localhost:6006/
