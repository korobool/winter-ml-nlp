from __future__ import absolute_import
from __future__ import print_function


import os
import h5py
import json
import shutil
import codecs
import yaml
import pylab as plt
import uuid
import string
import numpy as np
np.random.seed(1337)  # for reproducibility

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from pathlib import Path
from matplotlib import cm
# from keras.models import Sequential
# from keras.layers.core import Dense, Dropout, Activation
# from keras.optimizers import RMSprop
# from keras.models import model_from_json


class ModelNotTrainedError(Exception):
    pass


class MetricsNotCountedError(Exception):
    pass


class WrongMetricError(Exception):
    pass


class TokenizationMethodError(Exception):
    pass


class CategorizerTrainer:
    def __init__(self):

        # Version of product
        self._version = 1.63

        # Model id
        self._model_id = None

        # Model path
        self._model_path = None

        # Dictionary of models
        self._models = {}

        # Dictionary of metrics
        self._metrics = {}

        # List of categories
        self._categories = None

        # List of methods
        self._methods = None

        # Letters
        self._letters = False

        # Words
        self._words = False

        # Words_raw
        self._words_raw = False

        # Bigrams and trigrams
        self._ngrams = False

        # Symbolic trigrams
        self._symbolic_ngrams = None

        # Inflicted words
        self._inflicted_words_path = None

        # Tokenization type
        self._tokenization = None

    def fit(self, prep_data, methods, categories, features):
        """
        Performs model training on given data.
        :param prep_data: train data prepared with Preprocessor
        :param methods: list of methods for model making
                        ('SVC' or 'MultinomialNB')
        :param categories: list of train data categories
        :param features: list of model features
        """

        self._categories = categories
        self._methods = methods

        one_feature = False

        if 'letters' in features:
            self._letters = True
            one_feature = True

        if 'words' in features:
            self._words = True
            one_feature = True

        if 'words_raw' in features:
            self._words_raw = True
            self._words = True
            one_feature = True

        if 'ngrams' in features:
            self._ngrams = True
            one_feature = True

        if 'sym_ngrams_window' in features:
            self._symbolic_ngrams = 'window'
            one_feature = True

        if 'sym_ngrams_simple' in features:
            self._symbolic_ngrams = 'simple'
            one_feature = True

        if not one_feature:
            self._words = True

        if prep_data['is_data_balanced'] == True:
            self._models['_models_'] = {}
            for method in self._methods:
                self._models['_models_'][method] = {}
            for data in prep_data['data']:
                category_one = [categories[prep_data['data'].index(data)]]
                inflicted_words = []
                if self._inflicted_words_path:
                    file_path = Path(self._inflicted_words_path)
                    file_name = category_one[0] + '.txt'
                    if os.path.isfile(str(file_path / file_name)):
                        with codecs.open(str(file_path / file_name),
                                         encoding='utf-8', mode='r') as fp:
                            _inflicted_words = fp.readlines()
                            inflicted_words = [x.strip('\n') for x in _inflicted_words]
                dict_model = train(data, methods, category_one,
                                   self._letters, self._words,
                                   self._words_raw, self._ngrams,
                                   self._symbolic_ngrams,
                                   inflicted_words)
                for method in dict_model:
                    self._models['_models_'][method].update(dict_model[method])
        else:
            inflicted_words = []
            if self._inflicted_words_path:
                file_path = Path(self._inflicted_words_path)
                if os.path.isfile(str(file_path / 'all.txt')):
                    with codecs.open(str(file_path / 'all.txt'),
                                         encoding='utf-8', mode='r') as fp:
                        _inflicted_words = fp.readlines()
                        inflicted_words = [x.strip('\n') for x in _inflicted_words]
            self._models['_models_'] = train(prep_data['data'],
                                             methods,
                                             categories,
                                             self._letters,
                                             self._words,
                                             self._words_raw,
                                             self._ngrams,
                                             self._symbolic_ngrams,
                                             inflicted_words)

        self._model_id = str(uuid.uuid4())
        self._tokenization = prep_data['tokenization']
        self._models['_model_info_'] = \
            dict(methods=self._methods,
                 ngrams=self._ngrams,
                 symbolic_ngrams=self._symbolic_ngrams,
                 letters=self._letters,
                 words=self._words,
                 words_raw=self._words_raw,
                 categories=categories,
                 version=self._version,
                 model_id=self._model_id,
                 tokenization=self._tokenization)

    def save_models(self, model_path):
        """
        Save previously fitted models.
        :param model_path: path to save the models
        """

        self._model_path = model_path
        if not self._models:
            raise ModelNotTrainedError('Before save_models'
                                       ' CategorizerTrainer should be trained'
                                       ' (fit).')
        _dir_remove(model_path)
        model_path = Path(model_path)
        for method in self._methods:
            _dir_create(str(model_path / method))
            for category in self._models['_models_'][method]:
                _dir_create(str(model_path / method / category))
                if method in {'SVC', 'MultinomialNB'}:
                    joblib.dump(self._models['_models_'][method][category][0],
                                str(model_path /
                                    method /
                                    category /
                                    'model.pkl'))
                if method == 'Keras':
                    file_path = str(model_path /
                                    method /
                                    category /
                                    'model.json')
                    json_string = self._models['_models_'][method][category][0].to_json()
                    open(file_path, 'w').write(json_string)
                    file_path = str(model_path /
                                    method /
                                    category /
                                    'model.h5')
                    self._models['_models_'][method][category][0].save_weights(file_path)
                file_path = str(model_path /
                                method /
                                category /
                                'words.txt')
                with codecs.open(file_path, encoding='utf-8', mode='w') as fp:
                    for x in self._models['_models_'][method][category][1]:
                        fp.write("%s\n" % x)
        file_path = str(model_path / 'info.yaml')
        with open(file_path, 'w') as info_yaml:
            info_yaml.write(yaml.dump(
                self._models['_model_info_'],
                default_flow_style=False, allow_unicode=True))

    def evaluate_models(self, prep_data, **kwargs):
        """
        Evaluate previously loaded or fitted models.
        :param prep_data: test data prepared with Preprocessor
        :param kwargs: optional parameter save_metrics=True for saving metrics
        """

        if not self._models:
            raise ModelNotTrainedError('Before evaluate_models'
                                       ' CategorizerTrainer should be trained'
                                       ' (fit) or model should be loaded'
                                       ' (load_models).')


        if self._tokenization != prep_data['tokenization']:
            raise TokenizationMethodError('Tokenization method for test data'
                                          ' should be the same as for model.')

        if prep_data['is_data_balanced'] == True:
            self._metrics = {}
            for method in self._methods:
                self._metrics[method] = {}
            for data in prep_data['data']:
                category_one = \
                    [self._categories[prep_data['data'].index(data)]]
                dict_model = \
                    test(data, self._methods, category_one, self._models)
                for method in dict_model:
                    self._metrics[method].update(dict_model[method])
        else:
            self._metrics = test(prep_data['data'], self._methods,
                                 self._categories, self._models)
        if 'save_metrics' in kwargs:
            if kwargs['save_metrics'] == True:
                model_path = Path(self._model_path)
                for method in self._methods:
                    file_path = str(model_path / method / 'metrics.yaml')
                    with open(file_path, 'w') as metrics_yaml:
                        metrics_yaml.write(yaml.dump(self._metrics[method],
                                                     default_flow_style=False,
                                                     allow_unicode=True))

    def load_models(self, model_path):
        """
        Load previously fitted and saved models.
        :param model_path: path to dir with models
        """

        self._model_path = model_path
        self._models = {}
        methods = [d for d in os.listdir(model_path)
                   if os.path.isdir(os.path.join(model_path, d))]
        self._methods = methods
        model_path = Path(model_path)
        self._models['_models_'] = {}

        with open(str(model_path / 'info.yaml'), 'r') as fp:
            model_info = yaml.load(fp)
        self._models['_model_info_'] = model_info

        for method in methods:
            self._categories = model_info['categories']
            self._words = model_info['words']
            self._words_raw = model_info['words_raw']
            self._ngrams = model_info['ngrams']
            self._symbolic_ngrams = model_info['symbolic_ngrams']
            self._tokenization = model_info['tokenization']
            category_model = {}
            if method in {'SVC', 'MultinomialNB'}:
                for category in self._categories:
                    file_path = model_path / method / category
                    with codecs.open(str(file_path / 'words.txt'),
                                     encoding='utf-8', mode='r') as fp:
                        _words = fp.readlines()
                        words = [x.strip('\n') for x in _words]
                    category_model[category] = \
                        (joblib.load(str(file_path / 'model.pkl')), words)

            if method == 'Keras':
                for category in self._categories:
                    file_path = model_path / method / category
                    with codecs.open(str(file_path / 'words.txt'),
                                     encoding='utf-8', mode='r') as fp:
                        _words = fp.readlines()
                        words = [x.strip('\n') for x in _words]
                    file_path = \
                        str(model_path / method / category / 'model.json')
                    model = \
                        model_from_json(open(file_path).read())
                    file_path = \
                        str(model_path / method / category / 'model.h5')
                    model.load_weights(file_path)
                    category_model[category] = (model, words)

            self._models['_models_'][method] = category_model
            if os.path.isfile(str(model_path / method / 'metrics.yaml')):
                with open(str(model_path / method / 'metrics.yaml'), 'r') as fp:
                    self._metrics[method] = yaml.load(fp)

    def print_metrics(self, **kwargs):
        """
        Print the metrics previously calculated.
        :param kwargs: optional parameter - list of categories for printing
        """

        if not self._metrics:
            raise MetricsNotCountedError('Before print_metrics'
                                         ' CategorizerTrainer should count'
                                         ' metrics (evaluate_models).')
        for method in self._metrics:
            print('Model method:', method)
            categories = self._metrics[method]
            if 'categories' in kwargs:
                if set(self._metrics[method]).issuperset(set(kwargs['categories'])):
                    categories = kwargs['categories']
            for category in categories:
                print('Category:', category)
                metric_str = ''
                for metric in self._metrics[method][category]:
                    metrics_dic = self._metrics[method][category]
                    metric_str = \
                        ' '.join([metric_str, metric + '='
                                  + str(metrics_dic[metric])])
                print(metric_str.strip())
            print('\n')

    def set_inflicted_words(self, inflicted_words_path):
        """
        Set inflicted words to the vector.
        :param inflicted_words: list of inflicted words
        """
        self._inflicted_words_path = inflicted_words_path

    def draw_metrics(self, metric, **kwargs):
        """
        Draw histograms on the metrics previously calculated.
        :param metric: name of metric for drawing (pr, rc, f1, etc.)
        :param kwargs: optional parameter - list of categories for drawing
        """

        if not self._metrics:
            raise MetricsNotCountedError('Before draw_metrics'
                                         ' CategorizerTrainer should count'
                                         ' metrics (evaluate_models).')

        if metric not in ['pr', 'f1', 'rc', 'acc', 'correct', 'total']:
            raise WrongMetricError('Permitted metrics: pr, f1, rc, acc,'
                                   ' correct, total. Give correct metric'
                                   ' name.')
        categories = self._categories
        if 'categories' in kwargs:
                if set(self._categories).issuperset(set(kwargs['categories'])):
                    categories = kwargs['categories']
        pr_methods = []
        for method in self._methods:
            metrics_method = self._metrics[method]
            pr_method = []
            for category in categories:
                pr_method.append(metrics_method[category][metric])
            pr_methods.append(pr_method)

        fig = plt.figure(num=None,
                         figsize=(12, 8),
                         dpi=80,
                         facecolor='w',
                         edgecolor='k')
        ax = fig.add_subplot(111)
        N = len(categories)

        # necessary variables
        ind = np.arange(N)  # the x locations for the groups
        width = 0.2         # the width of the bars

        # the bars
        rects = []
        i = 0
        M = len(self._methods)
        for pr_method in pr_methods:
            rects.append(ax.bar(ind+width*i, pr_method, width,
                                color=cm.gist_rainbow(1.0*i/M)))
            i += 1

        # axes and labels
        ax.set_xlim(-width, len(ind)+width)
        ax.set_ylabel('Value of metric')
        ax.set_title(metric)
        ax.set_xticks(ind+width)
        xtickNames = ax.set_xticklabels(categories)
        plt.setp(xtickNames, rotation=45, fontsize=12)

        # add a legend
        rects_tuple = tuple([rect[0] for rect in rects])
        methods_tuple = tuple(self._methods)
        ax.legend(rects_tuple, methods_tuple, bbox_to_anchor=(1.25, 1))
        plt.show()


def draw_compare(categorizer1, categorizer2, metric):
    """
    Draw in comparison the metrics previously calculated.
    :param categorizer1: first CategorizerTrainer for comparison
    :param categorizer2: second CategorizerTrainer for comparison
    :param metric: name of metric for comparison
    """

    if metric not in ['pr', 'f1', 'rc', 'acc', 'correct', 'total']:
        raise WrongMetricError('Permitted metrics: pr, f1, rc, acc, correct,'
                               ' total. Give correct metric name.')

    metrics1 = categorizer1._metrics
    metrics2 = categorizer2._metrics
    metrics_SVC1 = metrics1['SVC']
    metrics_SVC2 = metrics2['SVC']
    metrics_MultinomialNB1 = metrics1['MultinomialNB']
    metrics_MultinomialNB2 = metrics2['MultinomialNB']
    pr_SVC1 = []
    pr_MultinomialNB1 = []
    pr_SVC2 = []
    pr_MultinomialNB2 = []
    categories = []

    for category in metrics_SVC1:
            pr_SVC1.append(metrics_SVC1[category][metric])
            pr_SVC2.append(metrics_SVC2[category][metric])
            pr_MultinomialNB1.append(metrics_MultinomialNB1[category][metric])
            pr_MultinomialNB2.append(metrics_MultinomialNB2[category][metric])
            categories.append(category)

    fig = plt.figure(num=None, figsize=(14, 10),
                     dpi=80, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(111)
    N = len(metrics_SVC1)

    # necessary variables
    ind = np.arange(N)  # the x locations for the groups
    width = 0.1         # the width of the bars

    # the bars
    rects1 = ax.bar(ind, pr_SVC1, width, color='blue')
    rects2 = ax.bar(ind+width, pr_SVC2, width, color='yellow')
    rects3 = ax.bar(ind+width*2, pr_MultinomialNB1, width, color='green')
    rects4 = ax.bar(ind+width*3, pr_MultinomialNB2, width, color='red')

    # axes and labels
    ax.set_xlim(-width, len(ind)+width)
    ax.set_ylabel('Value of metric')
    ax.set_title(metric)
    ax.set_xticks(ind+width)
    xtickNames = ax.set_xticklabels(categories)
    plt.setp(xtickNames, rotation=45, fontsize=12)

    # add a legend
    ax.legend((rects1[0], rects2[0], rects3[0], rects4[0]),
              ('SVC 1', 'SVC 2', 'MultinomialNB 1', 'MultinomialNB 2'),
              bbox_to_anchor=(1.2, 1))
    plt.show()


def draw_compare2(categorizers, categories, metric):
    """
    Draw in comparison the metrics previously calculated.
    :param categorizers: list of CategorizerTrainers
    :param categories: list of categories for drawing
    :param metric: name of metric for comparison
    """

    if metric not in ['pr', 'f1', 'rc', 'acc', 'correct', 'total']:
        raise WrongMetricError('Permitted metrics: pr, f1, rc, acc, correct,'
                               ' total. Give correct metric name.')
    metrics_SVC = []
    metrics_MultNB = []
    metrics_Keras = []
    for categorizer in categorizers:
        one_categorizer_SVC = []
        one_categorizer_MultNB = []
        one_categorizer_Keras = []
        for category in categories:
            one_categorizer_SVC.append(
                categorizer._metrics['SVC'][category][metric])
            one_categorizer_MultNB.append(
                categorizer._metrics['MultinomialNB'][category][metric])
            one_categorizer_Keras.append(
                categorizer._metrics['Keras'][category][metric])
        metrics_SVC.append(one_categorizer_SVC)
        metrics_MultNB.append(one_categorizer_MultNB)
        metrics_Keras.append(one_categorizer_Keras)

    fig = plt.figure(num=None,
                     figsize=(14, 10),
                     dpi=80,
                     facecolor='w',
                     edgecolor='k')
    ax = fig.add_subplot(111)
    N = len(categories)

    # necessary variables
    ind = np.arange(N)  # the x locations for the groups
    width = 0.2/len(categories)   # the width of the bars

    # the bars
    rects_SVC = []
    rects_MultNB = []
    rects_Keras = []
    for i in range(len(categorizers)):
        rects_SVC.append(ax.bar(ind+(width*i*3),
                                metrics_SVC[i], width,
                                color=cm.Paired(1.0*i/len(categorizers))))
        rects_MultNB.append(ax.bar(ind+(width*(3*i+1)),
                                   metrics_MultNB[i], width,
                                   color=cm.Dark2(1.0*i/len(categorizers))))
        rects_Keras.append(ax.bar(ind+(width*(3*i+2)),
                                  metrics_Keras[i], width,
                                  color=cm.Dark2(1.0*i/len(categorizers))))

    # axes and labels
    ax.set_xlim(-width, len(ind)+width)
    ax.set_ylabel('Value of metric')
    ax.set_title(metric)
    ax.set_xticks(ind+width*6)
    xtickNames = ax.set_xticklabels(categories)
    plt.setp(xtickNames, rotation=30, fontsize=12)

    # add a legend
    rects_list = []
    info_list = []
    for i in range(len(categorizers)):
        rects_list.append(rects_SVC[i][0])
        rects_list.append(rects_MultNB[i][0])
        rects_list.append(rects_Keras[i][0])
        info_list.append('model'+str(i+1)+' SVC')
        info_list.append('model'+str(i+1)+' MultNB')
        info_list.append('model'+str(i+1)+' Keras')
    rects_tuple = tuple(rects_list)
    info_tuple = tuple(info_list)

    ax.legend(rects_tuple, info_tuple, bbox_to_anchor=(1.2, 1))
    plt.show()


def _dir_remove(path):
    """
    Remove dir with old models and create empty dir for new models.
    :param path: path to dir
    """

    if (os.path.exists(path)) and (os.listdir(path) != []):
        shutil.rmtree(path)
        os.makedirs(path)


def _dir_create(path):
    """
    Create dir if it is not exists.
    :param path: path to dir.
    """

    if not os.path.exists(path):
        os.makedirs(path)


def keras(X_train, Y_train):
    """
    Make keras model on train data.
    :param X_train: vector of train data
    :param Y_train: vector of test data
    :param categories: list of categories
    :return: fitted keras model
    """

    Y_train = [([1, 0] if y == 1 else [0, 1]) for y in Y_train]
    Y_train = np.asarray(Y_train)

    X_train = X_train.astype("float32")
    Y_train = Y_train.astype("int8")

    # Number of epochs
    nb_epoch = 10

    # Number of classes (number of categories)
    nb_classes = 2

    model = Sequential()

    # Dense(128) is a fully-connected layer with 128 hidden units.
    model.add(Dense(128, input_shape=(len(X_train[0]),)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(nb_classes))

    # Softmax applied across inputs last dimension.
    model.add(Activation('softmax'))

    rms = RMSprop()
    model.compile(loss='binary_crossentropy',
                  optimizer=rms,
                  class_mode='binary')

    model.fit(X_train, Y_train,
              nb_epoch=nb_epoch,
              batch_size=128,
              verbose=1,
              show_accuracy=True,
              validation_split=0.1)
    return model


def alter_analyzer_words(_dict_one):
    """
    Extracting the sequence of features for CountVectorizer,
    that makes a bag of words from list of words.
    :param _dict_one: dictionary with data. For example:
            {'labels': [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            'ngrams': ['aghhhhhh accident', 'accident ha',
            'ha shut', 'shut hazmat', 'hazmat spill', 'spill clean',
            'clean ride', 'ride home', 'home goning', 'goning suck',
            'aghhhhhh accident ha', 'accident ha shut', 'ha shut hazmat',
            'shut hazmat spill', 'hazmat spill clean', 'spill clean ride',
            'clean ride home', 'ride home goning', 'home goning suck'],
            'symbolic_ngrams': ['agh', 'ghh', 'hhh', 'hhh', 'hhh', 'hhh',
            'acc', 'cci', 'cid', 'ide', 'den', 'ent', 'shu', 'hut', 'haz',
            'azm', 'zma', 'mat', 'spi', 'pil', 'ill', 'cle', 'lea', 'ean',
            'rid', 'ide', 'hom', 'ome', 'gon', 'oni', 'nin', 'ing', 'suc',
            'uck'], 'words': ['aghhhhhh', 'accident', 'ha', 'shut',
            'hazmat', 'spill', 'clean', 'ride', 'home', 'goning', 'suck']}
    :return: words for making bag of words
    """

    for _word in _dict_one['words']:
        yield _word


def alter_analyzer_ngrams(_dict_one):
    """
    Extracting the sequence of features for CountVectorizer,
    that makes a bag of words from list of ngrams.
    :param _dict_one: dictionary with data
    :return: ngrams for making bag of words
    """

    for _ngram in _dict_one['ngrams']:
        yield _ngram


def alter_analyzer_sym_ngrams(_dict_one):
    """
    Extracting the sequence of features for CountVectorizer,
    that makes a bag of words from list of symbolic ngrams.
    :param _dict_one: dictionary with data
    :return:symbolic ngrams for making bag of words
    """

    for _symbolic_ngram in _dict_one['symbolic_ngrams']:
        yield _symbolic_ngram


def alter_analyzer_variate(_dict_one):
    """
    Extracting the sequence of features for TfidfVectorizer,
    that prepare data for training.
    :param _dict_one: dictionary with data
    :return: letters, words, ngrams, symbolic ngrams for TfidfVectorizer
    """

    if 'letters' in _dict_one:
        for _letter in _dict_one['letters']:
            yield _letter
    if 'words' in _dict_one:
        for _word in _dict_one['words']:
            yield _word
    if 'ngrams' in _dict_one:
        for _ngram in _dict_one['ngrams']:
            yield _ngram
    if 'symbolic_ngrams' in _dict_one:
        for _symbolic_ngram in _dict_one['symbolic_ngrams']:
            yield _symbolic_ngram


def train(prep_data, methods, categories, letters_,
          words_, words_raw_, ngrams_, symbolic_ngrams_,
          inflicted_words):
    """
    Train models on given data.
    :param prep_data: data for training models
    :param methods: list of methods
    :param categories: list of categories
    :param letters_: True if letters are involved in training models
    :param words_: True if normalized words are involved in training models
    :param words_raw_: True the raw words are involved in training models
    :param ngrams_: True if ngrams are involved in training models
    :param symbolic_ngrams_: sym_ngrams_window or sym_ngrams_simple
                             if symbolic ngrams are involved in training
                             models
    :return: trained models
    """
    # The defaults for min_df and max_df are 1 and 1.0, respectively.
    # This basically says "If my term is found in only 1 document,
    # then it's ignored. Similarly if it's found in all documents
    # (100% or 1.0) then it's ignored."

    # Making the bag of words
    words = []

    if words_:
        vectoriz_words = CountVectorizer(analyzer=alter_analyzer_words,
                                         min_df=0.0005, max_df=0.99)
        vectoriz_words.fit_transform(prep_data)
        words = words + vectoriz_words.get_feature_names()
        if inflicted_words:
            delta_words = set(inflicted_words) - set(words)
            words.extend(list(delta_words))

    if ngrams_:
        vectoriz_ngrams = CountVectorizer(analyzer=alter_analyzer_ngrams,
                                          min_df=0.0005, max_df=0.99)
        vectoriz_ngrams.fit_transform(prep_data)
        words = words + vectoriz_ngrams.get_feature_names()

    if symbolic_ngrams_ in ['simple', 'window']:
        vectoriz_sym_ngrams = CountVectorizer(analyzer=alter_analyzer_sym_ngrams,
                                              min_df=0.0005, max_df=0.99)
        vectoriz_sym_ngrams.fit_transform(prep_data)
        sym_trigrams = vectoriz_sym_ngrams.get_feature_names()
        for x in sym_trigrams:
            if x not in words:
                words.append(x)

    if letters_:
        words = words + list(string.ascii_lowercase)

    # Clearing list of words from digits
    words = [x for x in words if not x.isdigit()]

    cv = TfidfVectorizer(analyzer=alter_analyzer_variate,
                         vocabulary=words)
    vect_sent = cv.fit_transform(prep_data)

    _models = {}
    for method in methods:
        category_model = {}
        if method == 'Keras':
            for j in categories:
                label_one = [x['labels'][categories.index(j)] for x in prep_data]
                vect_label = np.array(label_one)
                category_model[j] = (keras(vect_sent.toarray(), vect_label),
                                     words)
        if method in {'SVC', 'MultinomialNB'}:
            for j in categories:
                if method == 'SVC':
                    clf = SVC(kernel='linear')
                if method == 'MultinomialNB':
                    clf = MultinomialNB()
                # Select the column of category by index in categories
                label_one = [x['labels'][categories.index(j)] for x in prep_data]
                vect_label = np.array(label_one)
                clf.fit(vect_sent, vect_label)
                category_model[j] = (clf, words)
        _models[method] = category_model
    return _models


def test(prep_data, methods, categories, models):
    """
    Test models on given data.
    :param prep_data: data for testing models
    :param methods: list of methods
    :param categories: list of categories
    :param models: models for testing
    :return: dictionary of metrics
    """
    metrics_all = {}
    for method in methods:
        metrics = {}
        for j in categories:
            words = models['_models_'][method][j][1]
            cv = TfidfVectorizer(analyzer=alter_analyzer_variate,
                                 vocabulary=words)
            vect_sent_test = cv.fit_transform(prep_data)

            clf = models['_models_'][method][j][0]
            label_one = \
                [x['labels'][categories.index(j)] for x in prep_data]
            label_test = []
            total = len(prep_data)
            for x in vect_sent_test:
                if method in {'SVC', 'MultinomialNB'}:
                    label_test.extend(clf.predict(x).tolist())
                if method == 'Keras':
                    sent_array = x.toarray()
                    prediction = clf.predict_classes(sent_array,
                                                     batch_size=128,
                                                     verbose=0)
                    prediction = [(1 if x[0] == 1 else 0) for x in prediction]
                    y = [int(d) for d in prediction]
                    label_test.extend(y)
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
            metrics[j] = {'pr': pr, 'f1': f1, 'acc': acc, 'rc': rc,
                          'correct': correct, 'total': total,
                          'confusion_matrix': conf_matrix}
        metrics_all[method] = metrics
    return metrics_all
