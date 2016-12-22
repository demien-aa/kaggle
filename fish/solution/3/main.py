import cv2
import glob
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

from common.util import submit_stamp, red, green, white, blue, single_line, picklable, print_header_footer
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.models import Sequential
from keras.optimizers import SGD, RMSprop
from keras.utils import np_utils
import pandas as pd
from sklearn.cross_validation import KFold
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder


FISH_CLASSES = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
WIDTH = 32
HEIGHT = 32
RESIZE = (WIDTH, HEIGHT)  # Most train images are 1280 * 720
CHANNELS = 3 # RGB
RELOAD = True


def main():
    x_all, y_all = load_training_images()
    models = cross_validation_create_models(x_all, y_all)
    cross_validation_predict_test(models)


@print_header_footer('Load Training Images')
@picklable(__file__, reload=False or RELOAD)
def load_training_images():
    x_all, y_all = [], []
    for fish in FISH_CLASSES:
        image_pathes = get_images_path(fish)
        y_all.extend(np.tile(fish, len(image_pathes)))
        print("%s photos of %s" % (len(image_pathes), fish))
        for image_path in image_pathes:
            x_all.append(read_image_cv(image_path))

    x_all = np.array(x_all) / 256
    y_all = np.array(y_all)

    y_all = LabelEncoder().fit_transform(y_all)
    y_all = np_utils.to_categorical(y_all)
    return x_all, y_all


@print_header_footer('Load Test Images')
@picklable(__file__, reload=False or RELOAD)
def load_test_images():
    y_all, image_names = [], []
    fold_path = os.path.join('data', 'test_stg1', '*.jpg')
    image_pathes = glob.glob(fold_path)
    for image_path in image_pathes:
        y_all.append(read_image_cv(image_path))
        image_names.append(os.path.basename(image_path))
    return np.array(y_all) / 256, image_names


@print_header_footer('Create Model by Croos Validation')
def cross_validation_create_models(x_all, y_all, n_folds=10):
    random_state = 51
    batch_size = 16
    nb_epoch = 30
    num_fold = 0
    sum_score = 0
    models = []

    k = KFold(len(y_all), n_folds=n_folds, shuffle=True, random_state=random_state)
    for train_index, validation_index in k:
        num_fold += 1
        x_train = x_all[train_index]
        y_train = y_all[train_index]
        x_validation = x_all[validation_index]
        y_validation = y_all[validation_index]

        print(blue('Start KFold number %s from %s' % (num_fold, n_folds)))
        callbacks = [EarlyStopping(monitor='val_loss', patience=3, verbose=0), ]
        model = create_model()
        model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, shuffle=True, verbose=1, validation_data=(x_validation, y_validation), callbacks=callbacks)
        predictions_valid = model.predict(x_validation, batch_size=batch_size, verbose=1)
        score = log_loss(y_validation, predictions_valid)
        print(green("\nValidation Log Loss: %s" % score))
        models.append(model)
        sum_score += score * len(validation_index)

    score = sum_score / len(y_all)
    print(red("Log_loss train independent avg: %s" % score))
    return models


@print_header_footer('Predict test by Croos Validation')
def cross_validation_predict_test(models):
    batch_size = 16
    num_fold = 0
    yfull_test = []
    nfolds = len(models)
    test_data, image_names  = load_test_images()

    for i in range(nfolds):
        model = models[i]
        num_fold += 1
        print('Start KFold number {} from {}'.format(num_fold, nfolds))
        test_prediction = model.predict(test_data, batch_size=batch_size, verbose=2)
        yfull_test.append(test_prediction)

    test_res = merge_several_folds_mean(yfull_test, nfolds)
    create_submission(test_res, image_names)


def create_submission(predictions, test_id):
    result = pd.DataFrame(predictions, columns=['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT'])
    result.loc[:, 'image'] = pd.Series(test_id, index=result.index)
    sub_file = 'result/solution-3_%s.csv' % submit_stamp()
    result.to_csv(sub_file, index=False)


def merge_several_folds_mean(data, nfolds):
    a = np.array(data[0])
    for i in range(1, nfolds):
        a += np.array(data[i])
    a /= nfolds
    return a.tolist()


def get_images_path(fish):
    fold_path = os.path.join('data', 'train', fish, '*.jpg')
    return glob.glob(fold_path)


def read_image_cv(image_path):
    image = cv2.imread(image_path)
    resized = cv2.resize(image, RESIZE, cv2.INTER_LINEAR)
    return resized


def create_model():
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(HEIGHT, WIDTH, CHANNELS), dim_ordering='tf'))
    model.add(Convolution2D(4, 3, 3, activation='relu', dim_ordering='tf'))
    model.add(ZeroPadding2D((1, 1), dim_ordering='tf'))
    model.add(Convolution2D(4, 3, 3, activation='relu', dim_ordering='tf'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='tf'))

    model.add(ZeroPadding2D((1, 1), dim_ordering='tf'))
    model.add(Convolution2D(8, 3, 3, activation='relu', dim_ordering='tf'))
    model.add(ZeroPadding2D((1, 1), dim_ordering='tf'))
    model.add(Convolution2D(8, 3, 3, activation='relu', dim_ordering='tf'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='tf'))

    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(8, activation='softmax'))

    sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')
    # def center_normalize(x):
    #     return (x - K.mean(x)) / K.std(x)
    # model = Sequential()
    # model.add(Activation(activation=center_normalize, input_shape=(HEIGHT, WIDTH, CHANNELS)))
    # model.add(Convolution2D(32, 5, 5, border_mode='same', activation='relu', dim_ordering='tf'))
    # model.add(Convolution2D(32, 5, 5, border_mode='same', activation='relu', dim_ordering='tf'))
    # model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='tf'))
    # model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu', dim_ordering='tf'))
    # model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu', dim_ordering='tf'))
    # model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='tf'))
    # model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu', dim_ordering='tf'))
    # model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu', dim_ordering='tf'))
    # model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='tf'))
    # model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu', dim_ordering='tf'))
    # model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu', dim_ordering='tf'))
    # model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='tf'))
    # model.add(Flatten())
    # model.add(Dense(256, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(64, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(len(FISH_CLASSES)))
    # model.add(Activation('sigmoid'))
    # model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=1e-4))
    return model


if __name__ == '__main__':
    main()
