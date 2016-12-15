from common.util import submit_stamp
import time
import os, cv2, random, sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns
# %matplotlib inline 

from keras.models import Sequential
from keras.layers import Dropout, Flatten, Convolution2D, MaxPooling2D, ZeroPadding2D, Dense, Activation
from keras.optimizers import RMSprop, Adam
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from keras import backend as K


TRAIN_DIR = 'data/train/'
TEST_DIR = 'data/test_stg1/'
FISH_CLASSES = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
ROWS = 90  #720
COLS = 160 #1280
CHANNELS = 3 #RGB


def get_images(fish):
    """Load files from train folder"""
    fish_dir = TRAIN_DIR+'{}'.format(fish)
    images = [fish+'/'+im for im in os.listdir(fish_dir) if im != '.DS_Store']
    return images


def read_image(src):
    """Read and resize individual images"""
    im = cv2.imread(src, cv2.IMREAD_COLOR)
    im = cv2.resize(im, (COLS, ROWS), interpolation=cv2.INTER_CUBIC)
    return im


def load_images():
    files, y_all = [], []
    for fish in FISH_CLASSES:
        fish_files = get_images(fish)
        files.extend(fish_files)
        y_all.extend(np.tile(fish, len(fish_files)))
        print("{0} photos of {1}".format(len(fish_files), fish))

    y_all = np.array(y_all)
    x_all = np.ndarray((len(files), ROWS, COLS, CHANNELS), dtype=np.uint8)

    for i, im in enumerate(files):
        x_all[i] = read_image(TRAIN_DIR+im)
        if i%1000 == 0:
            print('Processed {} of {}'.format(i, len(files)))
    return x_all, y_all


def format_train_valid(x_all, y_all):
    y_all = LabelEncoder().fit_transform(y_all)
    y_all = np_utils.to_categorical(y_all)
    return train_test_split(x_all, y_all, test_size=0.2, random_state=23, stratify=y_all)


def train(x_train, y_train):
    optimizer = RMSprop(lr=1e-4)
    objective = 'categorical_crossentropy'

    def center_normalize(x):
        return (x - K.mean(x)) / K.std(x)

    model = Sequential()

    model.add(Activation(activation=center_normalize, input_shape=(ROWS, COLS, CHANNELS)))

    model.add(Convolution2D(32, 5, 5, border_mode='same', activation='relu', dim_ordering='tf'))
    model.add(Convolution2D(32, 5, 5, border_mode='same', activation='relu', dim_ordering='tf'))
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='tf'))

    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu', dim_ordering='tf'))
    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu', dim_ordering='tf'))
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='tf'))

    model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu', dim_ordering='tf'))
    model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu', dim_ordering='tf'))
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='tf'))

    model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu', dim_ordering='tf'))
    model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu', dim_ordering='tf'))
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='tf'))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(len(FISH_CLASSES)))
    model.add(Activation('sigmoid'))

    model.compile(loss=objective, optimizer=optimizer)

    early_stopping = EarlyStopping(monitor='val_loss', patience=4, verbose=1, mode='auto')
        
    model.fit(x_train, y_train, batch_size=64, nb_epoch=1, validation_split=0.2, verbose=1, shuffle=True, callbacks=[early_stopping])

    return model


def validate(model, x_valid, y_valid):
    preds = model.predict(x_valid, verbose=1)
    print("Validation Log Loss: {}".format(log_loss(y_valid, preds)))


def predict(model):
    test_files = [im for im in os.listdir(TEST_DIR) if im != '.DS_Store']
    test = np.ndarray((len(test_files), ROWS, COLS, CHANNELS), dtype=np.uint8)

    for i, im in enumerate(test_files): 
        test[i] = read_image(TEST_DIR+im)
    test_preds = model.predict(test, verbose=1)

    submission = pd.DataFrame(test_preds, columns=FISH_CLASSES)
    submission.insert(0, 'image', test_files)
    submission.head()

    submission.to_csv('result/solution-1_keras_%s.csv' % submit_stamp(), index=False)


if __name__ == '__main__':
    x_all, y_all = load_images()
    x_train, x_valid, y_train, y_valid = format_train_valid(x_all, y_all)
    model = train(x_train, y_train)
    validate(model, x_valid, y_valid)
    predict(model)
