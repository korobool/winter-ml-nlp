from __future__ import print_function
import numpy as np
import os
import h5py
import codecs
import re
import shutil
import csv
import sys
import glob
import yaml
import config

from keras.models import model_from_json
from pathlib import Path
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense, Activation, Reshape, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras.callbacks import Callback
from keras.optimizers import RMSprop
from keras.optimizers import SGD

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.acces = []
        self.yamls = []

    def on_epoch_end(self, epoch, logs={}):
        epoch_info = {'epoch': int(epoch+1),
                      'val_acc': float(logs.get('val_acc')),
                      'val_loss': float(logs.get('val_loss')),
                      'acc': self.acces,
                      'loss': self.losses}
        file_path = '/'.join(['./logs', str(epoch+1).zfill(4) + '.yaml'])
        with open(file_path, 'w') as info_yaml:
            info_yaml.write(yaml.dump(epoch_info,
                                      default_flow_style=False,
                                      allow_unicode=True))
        self.losses = []
        self.acces = []
        self.yamls.append(file_path)

    def on_batch_end(self, batch, logs={}):
        self.losses.append(float(logs.get('loss')))
        self.acces.append(float(logs.get('acc')))


class CLNet():
    def __init__(self):

        # Model
        self._model = None
        self._characters = config.characters
        self._batch = config.batch
        self._epoch = config.epoch
        self.max_num = None

    def fit(self, train_data_path, test_data_path, samples_per_epoch):

        # set parameters:
        batch_size = self._batch
        nb_epoch = self._epoch

        test_seq = _get_train_data(test_data_path)
        X_test = test_seq[0]
        y_test = test_seq[1]

        self.max_num = int(y_test.shape[1])

        print('Build model...')

        model = Sequential()

        # Alphabet x 1014
        model.add(Convolution2D(256, len(self._characters), 3, input_shape=(1, len(self._characters), 350)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(1, 3)))

        # 256 x 1 X  336
        model.add(Convolution2D(256, 1, 3))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(1, 3))) # MaxPooling missed in recommendations crepe.py

        # 110 x 256
        model.add(Convolution2D(256, 1, 3))
        model.add(Activation('relu'))

        # 108 x 256
        model.add(Convolution2D(256, 1, 3))
        model.add(Activation('relu'))

        # 106 x 256
        model.add(Convolution2D(256, 1, 3))
        model.add(Activation('relu'))

        # 104 x 256
        model.add(Convolution2D(256, 1, 3))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(1, 3)))

        # 34 x 256
        model.add(Flatten())

        # 8704
        model.add(Dense(1024, input_dim=8704))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        # 1024
        model.add(Dense(1024, input_dim=1024))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        # 1024
        model.add(Dense(self.max_num, input_dim=1024))
        # The loss (keras has no LogSoftmax so we have to improvise, don't forget calculate logarithms on test!)
        model.add(Activation('sigmoid'))

        opt = RMSprop()
        # opt = Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        # opt = Adam()
        # opt = SGD(lr=0.0001, momentum=0.9, decay=0.0005, nesterov=True)

        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=["accuracy"])
        checkpointer = ModelCheckpoint(filepath="./weights_tmp.h5", verbose=1, save_best_only=False)
        history = LossHistory()
        model.fit_generator(generator(train_data_path, batch_size),
                            nb_epoch=nb_epoch,
                            samples_per_epoch=samples_per_epoch,
                            validation_data=(X_test, y_test),
                            callbacks=[checkpointer, history])

        self._model = model
        with open('./logs/list.yaml', 'w') as info_yaml:
            info_yaml.write(yaml.dump(history.yamls,
                                      default_flow_style=False,
                                      allow_unicode=True))

    def save_model(self, model_path):
        # _dir_remove(model_path)
        model_path = Path(model_path)

        file_path = str(model_path / 'model.json')
        json_string = self._model.to_json()
        open(file_path, 'w').write(json_string)

        file_path = str(model_path / 'model.h5')
        self._model.save_weights(file_path)

    def load_model(self, model_path):
        model_path = Path(model_path)
        file_path = str(model_path / 'model.json')
        self._model = model_from_json(open(file_path).read())
        file_path = str(model_path / 'model.h5')
        self._model.load_weights(file_path)

    def test(self, sentence):
        X = np.empty((0, 1, len(self._characters), 350), np.int8)
        frame = np.zeros((len(self._characters), 350), dtype=np.int8)
        indexes = list(range(len(self._characters)))
        couples = dict(zip(self._characters, indexes))
        sentence = sentence.lower()
        if len(sentence) < 350:
            len_sent = len(sentence)
        else:
            len_sent = 350

        for i, ch in enumerate(sentence[:len_sent]):
            try:
                index = couples[ch]
                frame[index][i] = 1
            except KeyError:
                continue

        nb_1 = frame.shape[0]
        nb_2 = frame.shape[1]
        newshape = (1, 1, nb_1, nb_2)
        frame_list = np.reshape(frame, newshape)
        X = np.append(X, frame_list, axis=0)
        # nb_samples = X.shape[0]
        # nb_features1 = X.shape[1]
        # nb_features2 = X.shape[2]
        # newshape = (nb_samples, 1, nb_features1, nb_features2)
        # X = np.reshape(X, newshape)
        prediction = self._model.predict(X,
                                         batch_size=self._batch,
                                         verbose=0)
        label_test = prediction.tolist()
        for one_label in label_test:
            max_label = max(one_label)
            one_label_norm = [x/max_label for x in one_label]
            one_label_norm_ = []
            for y in one_label_norm:
                if y >= 0.8:
                    one_label_norm_.append(1)
                else:
                    one_label_norm_.append(0)
            label_test[label_test.index(one_label)] = one_label_norm_
        return label_test

    def evaluate_model(self, test_data_path):
        print('For test data in evaluation:')
        test_seq = _get_train_data(test_data_path)
        X_test = test_seq[0]
        y_test = test_seq[1]

        # nb_samples = X_test.shape[0]
        # nb_features1 = X_test.shape[1]
        # nb_features2 = X_test.shape[2]
        # newshape = (nb_samples, 1, nb_features1, nb_features2)
        # X_test = np.reshape(X_test, newshape)

        prediction = self._model.predict(X_test, batch_size=self._batch, verbose=0)
        # print('Prediction: ', prediction)
        label_test_all = prediction.tolist()
        for one_label in label_test_all:
            max_label = max(one_label)
            one_label_norm = [x/max_label for x in one_label]
            one_label_norm_ = []
            for y in one_label_norm:
                if y >= 0.8:
                    one_label_norm_.append(1)
                else:
                    one_label_norm_.append(0)
            label_test_all[label_test_all.index(one_label)] = one_label_norm_

        for i in range(self.max_num):
            print('Prediction for category ', i+1)

            label_test = [int(x[i]) for x in label_test_all]
            # print('Prediction:', label_test)

            label_one = [int(x[i]) for x in y_test]
            # print('Label:', label_one)

            total = len(label_test)
            pr = float(precision_score(label_one,
                                       label_test,
                                       average='binary'))
            rc = float(recall_score(label_one,
                                    label_test,
                                    average='binary'))
            f1 = float(f1_score(label_one,
                                label_test,
                                average='binary'))
            acc = float(accuracy_score(label_one,
                                       label_test))
            conf_matrix = confusion_matrix(label_one,
                                           label_test).tolist()
            correct = \
                int(accuracy_score(label_one, label_test, normalize=False))
            metrics = {'pr': pr, 'f1': f1, 'acc': acc, 'rc': rc,
                       'correct': correct, 'total': total,
                       'confusion_matrix': conf_matrix}
            print(metrics)


def _dir_remove(path):
    """
    Remove dir with old models and create empty dir for new models.
    :param path: path to dir
    """
    if (os.path.exists(path)) and (os.listdir(path) != []):
        shutil.rmtree(path)
        os.makedirs(path)
    else:
        os.makedirs(path)


def _get_train_data(file):
    print("Reading test files with tensor...")
    npzfile = np.load(file)
    X = npzfile['X']
    y = npzfile['y']
    return X, y


def generator(data_path, batch_size):
    #batch_index = 0
    files = glob.glob('/'.join([data_path, '*.npz']))
    while 1:
        for file in files:
            #file = next(files)
            #print(file)
            npzfile = np.load(file)
            X = npzfile['X']
            y = npzfile['y']
            yield (X, y)
        #N = X.shape[0]
        #current_index = (batch_index * batch_size) % N
        #if N >= current_index + batch_size:
        #    current_batch_size = batch_size
        #    batch_index += 1
        #else:
        #    current_batch_size = N - current_index
        #    batch_index = 0
        #yield (X[current_index: current_index + current_batch_size],
        #       y[current_index: current_index + current_batch_size])


if __name__ == '__main__':
    train = str(sys.argv[1])
    test = str(sys.argv[2])
    model = str(sys.argv[3])
    train_shape = int(sys.argv[4])
    net = CLNet()
    net.fit(train, test, train_shape)
    net.save_model(model)
    net.evaluate_model(test)
