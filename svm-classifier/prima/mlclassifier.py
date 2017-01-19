import os
import codecs
import nltk
import yaml
import numpy as np
import re

from nltk.corpus import stopwords
from nltk.util import ngrams as ng
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
from pathlib import Path
# from keras.models import model_from_json
from configparser import ConfigParser


class ModelNotLoadedError(Exception):
    pass


class ModelVersionError(Exception):
    pass


class ConfigFileError(Exception):
    pass


class Categorizer:
    def __init__(self):

        # Version of product
        self._version = 1.63

        # Model id
        self._model_id = None

        # Dictionary of models
        self._models = {}

        # List of categories
        self._categories = []

        # List of methods
        self._methods = []

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

        # List of stopwords
        self._stopWords = stopwords.words('english')

        # Tokenization method and list of permitted tags
        self._tokenization = None
        self._good_tags = None
        self._parser = None

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

        self._tokenization = model_info['tokenization']
        if self._tokenization == 'spacy':
            from spacy.en import English
            self._parser = English()

        if self._tokenization == 'nltk':
            from nltk.stem.wordnet import WordNetLemmatizer
            self._parser = WordNetLemmatizer()

        good_tags = {
            'nltk': ['NN', 'NNS', 'NNP', 'NNPS',
                     'VB', 'VBD', 'VBG', 'VBN',
                     'VBP', 'VBZ', 'RB', 'RBR',
                     'RBS', 'RP', 'JJ', 'JJR',
                     'JJS', 'MD'],
            'spacy':  ['VERB', 'NOUN', 'CONJ',
                       'ADV', 'ADP', 'ADJ']
        }

        self._good_tags = good_tags[self._tokenization]

        self._categories = model_info['categories']
        self._words = model_info['words']
        self._words_raw = model_info['words_raw']
        self._ngrams = model_info['ngrams']
        self._symbolic_ngrams = model_info['symbolic_ngrams']

        for method in methods:
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
                    file_path = str(model_path / method / category / 'model.json')
                    model = model_from_json(open(file_path).read())
                    file_path = str(model_path / method / category / 'model.h5')
                    model.load_weights(file_path)
                    category_model[category] = (model, words)

            self._models['_models_'][method] = category_model

    def classify(self, text, method):
        """
        Multi-category text classification.
        :param text: text for classification
        :param method: method for classification ('SVC', 'MultinomialNB',
                       'Keras')
        """

        if not self._models:
            raise ModelNotLoadedError('Before Categorizer classify'
                                      ' models should be loaded'
                                      ' (load_models).')
        sent_test = _prepare_data2(text, self._parser, self._tokenization)

        test_list_norm = _sent_learn_prep2(sent_test,
                                           self._stopWords,
                                           self._good_tags,
                                           self._letters,
                                           self._words,
                                           self._words_raw,
                                           self._ngrams,
                                           self._symbolic_ngrams)

        categories_true = []
        for j in self._categories:
            words = self._models['_models_'][method][j][1]
            cv = TfidfVectorizer(analyzer=alter_analyzer_variate,
                                 vocabulary=words)

            vect_sent_test = cv.fit_transform(test_list_norm)

            clf = self._models['_models_'][method][j][0]

            if method in {'SVC', 'MultinomialNB'}:
                if clf.predict(vect_sent_test[0]).tolist()[0] == 1:
                    categories_true.append(1)
                else:
                    categories_true.append(0)
            if method == 'Keras':
                sent_array = vect_sent_test[0].toarray()
                prediction = clf.predict_classes(sent_array,
                                                 batch_size=128,
                                                 verbose=0)
                prediction = [(1 if x[0] == 1 else 0) for x in prediction]
                y = [int(d) for d in prediction]
                if y[0] == 1:
                    categories_true.append(1)
                else:
                    categories_true.append(0)
        return categories_true

    def get_categories_names(self, vector):
        """
        Getting names of categories for vector, received by classifying.
        :param vector: vector of categories,
                       for example:[0, 0, 0, 1, 0, 1, 0, 0, 0, 1]
        :return: list of categories names
        """

        category_names = []
        for category in self._categories:
            if vector[self._categories.index(category)] == 1:
                category_names.append(category)
        return category_names

    def get_models_info(self):
        """
        Getting information about model.
        :return: list with information about loaded models
        """

        if not self._models:
            raise ModelNotLoadedError('Before Categorizer getModelInfo'
                                      ' models should be loaded'
                                      ' (load_models).')
        loaded_models = []
        for method in self._methods:
            loaded_models.append(self._models[method]['_model_info_'])
        return loaded_models


def _prepare_data2(text, parser, tokenization):
    """
    Data preparation for vectorization (for classify).
    :param text: text for classifying.
    :return: normalized pseudo-text
    """

    sent_learn_norm = []
    text = re.sub(r'[=+*#@^|~:]', ' ', text)
    text = re.sub(' +', ' ', text)

    if tokenization == 'nltk':
        text = nltk.word_tokenize(text)
        text_norm_list = [parser.lemmatize(x) for x in text]
        tokens = nltk.pos_tag(text)
        pos_tags_list = list(map(lambda token: token[1], tokens))

    if tokenization == 'spacy':
        parsedData = parser(text)
        text = [token.orth_ for token in parsedData]
        pos_tags_list = [token.pos_ for token in parsedData]
        text_norm_list = [token.lemma_ for token in parsedData]

    sent_learn_norm.append((tuple(zip(text_norm_list, pos_tags_list)),
                            tuple(zip(text, pos_tags_list))))
    return sent_learn_norm


def _sent_learn_prep2(sent_, stop_words, good_tags,
                      letters_, words_, words_raw_,
                      ngrams_, symbolic_ngrams_):
    """
    Filtering by the part of speech and normalized sentences formation
    (for classify).
    :param sent_: list of tuples with normal forms of words, part
                  of speech and labels, for example:
    :param stop_words: list of stop  words
    :param good_tags: list of permitted tags
    :param letters_: True if letters are involved in training models
    :param words_: True if normalized words are involved in training models
    :param words_raw_: True the raw words are involved in training models
    :param ngrams_: True if ngrams are involved in training models
    :param symbolic_ngrams_: sym_ngrams_window or sym_ngrams_simple
                             if symbolic ngrams are involved in training
                             models
    :return: list of dictionaries with data for classifying
    """

    sent_list = []
    for tuple_ in sent_:
        sent_dict = {}
        words_list = filter_words(tuple_[0], stop_words, good_tags)
        if words_:
            sent_dict['words'] = words_list
        if words_raw_:
            words_raw_list = filter_words(tuple_[1], stop_words, good_tags)
            words_raw_list = list(set(words_raw_list).difference(set(words_list)))
            sent_dict['words'].extend(words_raw_list)
        if ngrams_:
            sent_dict['ngrams'] = _ngrams_prep(words_list)
        if symbolic_ngrams_ == 'simple':
            sent_dict['symbolic_ngrams'] = _sym_ngrams_prep(words_list)
        if symbolic_ngrams_ == 'window':
            sent_dict['symbolic_ngrams'] = _sym_ngrams_prep_window(words_list)
        sent_list.append(sent_dict)
    return sent_list


def _ngrams_prep(_words_list):
    """
    Making bigrams and trigrams.
    :param _words_list: list of words for getting ngrams
    :return: list of ngrams
    """

    ngrams_list = list(ng(_words_list, 2)) + list(ng(_words_list, 3))
    ngrams_list = [' '.join(gram) for gram in ngrams_list]
    return ngrams_list


def _sym_ngrams_prep(_words_list):
    """
    Making simple symbolic trigrams.
    :param _words_list: list of words for getting symbolic trigrams
    :return: list of simple symbolic trigrams,
             for example input is: ['this', 'is', 'a', 'foo', 'sentences']
             Output will be something like this:
             ['thi', 'his', 'foo', 'sen', 'ent', 'nte', 'ten', 'enc', 'nce',
             'ces']
    """

    trigrams_list = []
    for _word in _words_list:
        _trigrams = list(nltk.trigrams(_word))
        if _trigrams:
            _trigrams = [''.join(x) for x in _trigrams]
            trigrams_list.extend(_trigrams)
    return trigrams_list


def _sym_ngrams_prep_window(_words_list):
    """
    Making symbolic trigrams, window size = 3
    :param _words_list: list of words for getting symbolic trigrams
    :return: list of window symbolic trigrams,
             for example input is: ['this', 'is', 'a', 'foo', 'sentences']
             Output will be something like this:
             ['++t', '+th', 'thi', 'his', 'is+', 's++', '++i', '+is', 'is+',
             's++', '++a', '+a+', 'a++', '++f', '+fo', 'foo', 'oo+', 'o++',
             '++s', '+se', 'sen', 'ent', 'nte', 'ten', 'enc', 'nce', 'ces',
             'es+', 's++']
    """

    trigrams_list = []
    for _word in _words_list:
        _trigrams = list(ng(_word, 3, pad_left=True, pad_right=True, pad_symbol='+'))
        if _trigrams:
            _trigrams = [''.join(x) for x in _trigrams]
            trigrams_list.extend(_trigrams)
    return trigrams_list


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


def filter_words(tuple_one, stop_words, good_tags):
    """
    Filtering words by pos-tags and no presence in list of stop words.
    :param tuple_one: one tuple of data (represents one training unit)
    :param stop_words: list of stop words
    :param good_tags: list of good tags
    :return: list of filtered words
    """

    words_list = []
    for x in tuple_one:
        if (x[1] in good_tags) and \
                (x[0] not in stop_words) and \
                (len(x[0]) > 1) and \
                (x[0][0] not in {'/', '\'', '\"', '\\',
                                 '@', '.', '{', '}', '(',
                                 ')', '`', '[', ']', '_',
                                 ',', '-', '+', '^', ':',
                                     ';', '*', '|', '~', '&',
                                 '#', '!', '$'}):
            words_list.append(x[0].lower())
    return words_list
