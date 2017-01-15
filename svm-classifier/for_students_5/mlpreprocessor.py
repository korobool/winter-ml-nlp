import nltk
import codecs
import re
import pylab as plt
import numpy as np
import random
import copy
import time

from nltk.corpus import stopwords
from nltk.util import ngrams as ng


class DataNotPreparedError(Exception):
    pass


class ConfigFileError(Exception):
    pass


class Preprocessor:
    def __init__(self, tokenization='nltk'):

        # Version of product
        self._version = 1.63

        # List of categories
        self._categories = []

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
        self._stopWords = get_stop_words()

        # Tokenization method and list of permitted tagsif wordList is None:
        if tokenization == 'spacy':
            self._tokenization = 'spacy'
        else:
            self._tokenization = 'nltk'

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

        if self._tokenization == 'spacy':
            from spacy.en import English
            self._parser = English()

        if self._tokenization == 'nltk':
            from nltk.stem.wordnet import WordNetLemmatizer
            self._parser = WordNetLemmatizer()

        # Prepared data
        self.data = None

        # Balanced data
        self.data_balanced = None

        # Vector of zero labeled data
        self.zero = None

        # Vector of one labeled data
        self.one = None

        # Vector of zero labeled balanced data
        self.zero_balanced = None

        # Vector of one labeled balanced data
        self.one_balanced = None

    def prepare_data(self, train_data_path, categories, features):
        """
        Data preparation for vectorization (for fit).
        :param train_data_path: path to train data
        :param categories: list of train data categories
        :param features: list of features
        """

        self._categories = categories
        n_label = len(categories)
        is_data_balanced = False
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

        labels = []
        sent_learn = []
        pattern = _get_pattern(n_label)
        with codecs.open(train_data_path, encoding='utf-8',
                         mode='r') as f:
            for line in f:
                label = re.findall(pattern, line)
                sentence = re.split(pattern, line)[0]
                if not label:
                    continue
                label = label[0].strip('\t\n')
                label = label.split()
                label = [_set_label(x) for x in label]
                labels.append(label)
                sentence = re.sub(r'[=+*#@^|~:]', ' ', sentence)
                sentence = re.sub(' +', ' ', sentence)
                sent_learn.append((sentence, label))
        sent_learn_norm = []
        for sent in sent_learn:
            if self._tokenization == 'nltk':
                text = nltk.word_tokenize(sent[0])
                text_norm_list = [self._parser.lemmatize(x) for x in text]
                tokens = nltk.pos_tag(text)
                pos_tags_list = list(map(lambda token: token[1], tokens))
            if self._tokenization == 'spacy':
                parsedData = self._parser(sent[0])
                text = [token.orth_ for token in parsedData]
                pos_tags_list = [token.pos_ for token in parsedData]
                text_norm_list = [token.lemma_ for token in parsedData]
            sent_learn_norm.append((tuple(zip(text_norm_list, pos_tags_list)),
                                    sent[1],
                                    tuple(zip(text, pos_tags_list))))
        _data = _sent_learn_prep(sent_learn_norm,
                                 self._stopWords,
                                 self._good_tags,
                                 self._letters,
                                 self._words,
                                 self._words_raw,
                                 self._ngrams,
                                 self._symbolic_ngrams)
        zero_counts = []
        one_counts = []
        for category in self._categories:
            label_one = [x['labels'][self._categories.index(category)]
                         for x in _data]
            one_ = 0
            zero_ = 0
            for i in label_one:
                if i == 1:
                    one_ += 1
                else:
                    zero_ += 1
            zero_counts.append(zero_)
            one_counts.append(one_)
        self.zero = zero_counts
        self.one = one_counts
        self.data = dict(data=_data,
                         zero=dict(zip(self._categories, self.zero)),
                         one=dict(zip(self._categories, self.one)),
                         is_data_balanced=is_data_balanced,
                         tokenization=self._tokenization)

    def balance_data(self):
        """
        Balancing data after prepare_data
        """
        if not self.data:
            raise DataNotPreparedError('Before balance_data'
                                       ' Preprocessor should prepare'
                                       ' data (prepare_data).')
        is_data_balanced = True
        choices_zero_count = []
        choices_one_count = []
        balanced_data = []
        for category in self._categories:
            one_list = []
            zero_list = []
            i = self._categories.index(category)

            if self.one[i] < self.zero[i]:
                n_choices = self.one[i]
            else:
                n_choices = self.zero[i]

            for x in self.data['data']:
                label_one = x['labels'][i]
                y = copy.copy(x)
                if label_one == 1:
                    y['labels'] = [1]
                    one_list.append(y)
                else:
                    y['labels'] = [0]
                    zero_list.append(y)

            choices_one = random.sample(one_list, n_choices)
            choices_zero = random.sample(zero_list, n_choices)

            choices_one_count.append(len(choices_one))
            choices_zero_count.append(len(choices_zero))

            balanced_data.append(sorted(choices_one + choices_zero,
                                        key=lambda k: random.random()))
        self.zero_balanced = choices_zero_count
        self.one_balanced = choices_one_count
        self.data_balanced = dict(data=balanced_data,
                                  zero=dict(zip(self._categories,
                                                self.zero_balanced)),
                                  one=dict(zip(self._categories,
                                               self.one_balanced)),
                                  is_data_balanced=is_data_balanced,
                                  tokenization=self._tokenization)

    def print_statistic(self, **kwargs):
        """
        Print statistic information for train data.
        """

        if not self.data:
            raise DataNotPreparedError('Before print_statistic'
                                       ' Preprocessor should prepare'
                                       ' data (prepare_data).')
        _zero = self.zero
        _one = self.one
        if 'data_balanced' in kwargs:
            if kwargs['data_balanced'] == True:
                if not self.data_balanced:
                    raise DataNotPreparedError('Before print_statistic'
                                               ' for balanced data'
                                               ' Preprocessor should prepare'
                                               ' data (prepare_data)'
                                               ' and balance data'
                                               ' (balance_data)')
                else:
                    _zero = self.zero_balanced
                    _one = self.one_balanced

        for category in self._categories:
            zero = _zero[self._categories.index(category)]
            one = _one[self._categories.index(category)]
            print("%s: zero=%d, one=%d " % (category, zero, one))

    def draw_statistic(self, **kwargs):
        """
        Draw histogram for statistic information.
        """

        if not self.data:
            raise DataNotPreparedError('Before print_statistic'
                                       ' Preprocessor should prepare'
                                       ' data (prepare_data).')
        _zero = self.zero
        _one = self.one
        if 'data_balanced' in kwargs:
            if kwargs['data_balanced'] == True:
                if not self.data_balanced:
                    raise DataNotPreparedError('Before print_statistic'
                                               ' for balanced data'
                                               ' Preprocessor should prepare'
                                               ' data (prepare_data)'
                                               ' and balance data'
                                               ' (balance_data)')
                else:
                    _zero = self.zero_balanced
                    _one = self.one_balanced

        fig = plt.figure(num=None, figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')
        ax = fig.add_subplot(111)

        N = len(self._categories)

        # necessary variables
        ind = np.arange(N)  # the x locations for the groups
        width = 0.3         # the width of the bars

        # the bars
        rects1 = ax.bar(ind, _zero, width, color='red')
        rects2 = ax.bar(ind+width, _one, width, color='green')

        # axes and labels
        ax.set_xlim(-width,len(ind)+width)
        ax.set_ylabel('Number of labels')
        ax.set_title('Number of ones and zeros in labels')
        ax.set_xticks(ind+width)
        xtickNames = ax.set_xticklabels(self._categories)
        plt.setp(xtickNames, rotation=45, fontsize=12)

        # add a legend
        ax.legend( (rects1[0], rects2[0]), ('Zero', 'One') )
        plt.show()


def _set_label(x):
    """
    Labels more than one accepted equal to one.
    :param x: raw label value
    :return: normalized label value
    """

    if int(x) >= 1:
        label_ = 1
    else:
        label_ = 0
    return label_


def _get_pattern(n):
    """
    Make pattern for different length of categories.
    :param n: number of labels (categories)
    :return: pattern depending on the number of categories
    """

    if n == 1:
        return '[\t\b][0-9][\n]'
    else:
        return '[\t\b]' + '[0-9][\t]' * (n - 1) + '[0-9][\n]'


def _sent_learn_prep(sent_, stop_words, good_tags, letters_,
                     words_, words_raw_, ngrams_, symbolic_ngrams_):
    """
    Filtering by the part of speech and normalized sentences formation
    (for fit and evaluate_models).
    :param sent_: list of tuples with normal forms of words, part
            of speech and labels, for example:
            [((('AGHHHHHH', 'NNP'), (',', ','), ('an', 'DT'),
            ('accident', 'NN'), ('on', 'IN'), ('95', 'CD'),
            ('ha', 'VBZ'), ('it', 'PRP'), ('shut', 'VBD'),
            ('down', 'IN'), ('for', 'IN'), ('a', 'DT'),
            ('hazmat', 'NN'), ('spill', 'NN'), ('and', 'CC'),
            ('clean', 'JJ'), ('up', 'RP'), ('...', ':'),
            ('the', 'DT'), ('ride', 'NN'), ('home', 'NN'),
            ('is', 'VBZ'), ('goning', 'VBG'), ('to', 'TO'),
            ('suck', 'VB'), ('!', '.'), ('!', '.'), ('!', '.'),
            ('!', '.'), (':', ':'), ('-', ':'), ('(', ':')),
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0]), ...;
    :param stop_words: list of stop  words
    :param good_tags: list of permitted tags
    :param letters_: True if letters are involved in training models
    :param words_: True if normalized words are involved in training models
    :param words_raw_: True the raw words are involved in training models
    :param ngrams_: True if ngrams are involved in training models
    :param symbolic_ngrams_: sym_ngrams_window or sym_ngrams_simple
                             if symbolic ngrams are involved in training
                             models
    :return: list of dictionaries with data for training or evaluating models,
             for example:
             [{'labels': [0, 1, 0, 0, 0, 0, 0], 'symbolic_ngrams': ['++t',
             '+to', 'ton', 'oni', 'nig', 'igh', 'ght', 'ht+', 't++',
             '++s', '+sa', 'sat', 'atu', 'tur', 'urd', 'rda', 'day',
             'ay+', 'y++', '++c', '+cl', 'cla', 'lar', 'ara', 'ra+',
             'a++', '++g', '+gi', 'giv', 'ivi', 'vin', 'ing', 'ng+',
             'g++', '++a', '+aw', 'awa', 'way', 'ay+', 'y++', '++s',
             '+sa', 'sat', 'atu', 'tur', 'urd', 'rda', 'day', 'ay+',
             'y++', '++c', '+cl', 'cla', 'lar', 'ara', 'ra+', 'a++',
             '++h', '+ha', 'hat', 'at+', 't++', '++b', '+br', 'bro',
             'roa', 'oad', 'adc', 'dca', 'cas', 'ast', 'sti', 'tin',
             'ing', 'ng+', 'g++', '++c', '+ch', 'cha', 'ham', 'amp',
             'mpi', 'pio', 'ion', 'on+', 'n++', '++l', '+le', 'lea',
             'eag', 'agu', 'gue', 'ue+', 'e++', '++f', '+fi', 'fin',
             'ina', 'nal', 'al+', 'l++', '++b', '+bi', 'big', 'ig+',
             'g++', '++s', '+sc', 'scr', 'cre', 'ree', 'een', 'en+',
             'n++', '++e', '+ea', 'ear', 'arl', 'rly', 'ly+', 'y++',
             '++k', '+ke', 'key', 'ey+', 'y++', '++s', '+sa', 'sat',
             'atu', 'tur', 'urd', 'rda', 'day', 'ays', 'ys+', 's++',
             '++c', '+ch', 'cha', 'ham', 'amp', 'mpi', 'pio', 'ion',
             'ons', 'ns+', 's++', '++h', '+ha', 'hat', 'ats', 'ts+',
             's++', '++a', '+as', 'ass', 'ss+', 's++'],
             'ngrams': ['tonight saturday', 'saturday clara', 'clara giving',
             'giving away', 'away saturday', 'saturday clara', 'clara hat',
             'hat broadcasting', 'broadcasting champion', 'champion league',
             'league final', 'final big', 'big screen', 'screen early',
             'early key', 'key saturdays', 'saturdays champions',
             'champions hats', 'hats ass', 'tonight saturday clara',
             'saturday clara giving', 'clara giving away',
             'giving away saturday', 'away saturday clara',
             'saturday clara hat', 'clara hat broadcasting',
             'hat broadcasting champion', 'broadcasting champion league',
             'champion league final', 'league final big', 'final big screen',
             'big screen early', 'screen early key', 'early key saturdays',
             'key saturdays champions', 'saturdays champions hats',
             'champions hats ass'], 'words': ['tonight', 'saturday',
             'clara', 'giving', 'away', 'saturday', 'clara', 'hat',
             'broadcasting', 'champion', 'league', 'final', 'big', 'screen',
             'early', 'key', 'saturdays', 'champions', 'hats', 'ass']},
              {'labels': [0, 0, 0, 0, 1, 0, 0], ...

    """

    sent_list = []
    for tuple_one in sent_:
        weight = tuple_one[1]
        sent_dict = {}
        words_list = filter_words(tuple_one[0], stop_words, good_tags)
        if words_:
            sent_dict['words'] = words_list
        if words_raw_:
            words_raw_list = filter_words(tuple_one[2], stop_words, good_tags)
            words_raw_list = list(set(words_raw_list).difference(set(words_list)))
            sent_dict['words'].extend(words_raw_list)
        if letters_:
            sent_dict['letters'] = _letters_prep(words_list)
        if ngrams_:
            sent_dict['ngrams'] = _ngrams_prep(words_list)
        if symbolic_ngrams_ == 'simple':
            sent_dict['symbolic_ngrams'] = _sym_ngrams_prep(words_list)
        if symbolic_ngrams_ == 'window':
            sent_dict['symbolic_ngrams'] = _sym_ngrams_prep_window(words_list)
        sent_dict['labels'] = weight
        sent_list.append(sent_dict)

    return sent_list


def _letters_prep(_words_list):
    """
    Making list of letters.
    :param _words_list: list of words for getting letters
    :return: list of letters
    """

    letters_list = []
    for _word in _words_list:
        for _letter in _word:
            letters_list.append(_letter)
    return letters_list


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


def get_stop_words():
    """
    Getting stop words for filtering.
    :return: list of stop words
    """

    stop_words = stopwords.words('english') + \
                 ["'d", "'ll", "'m", "'re", "'s",
                  "'ve", '..', '...', '.i', "'v",
                  "'t"]
    stop_words = set(stop_words)
    return stop_words


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
                                 '#', '!', '$', '1', '2', '3',
                                 '4', '5', '6', '7', '8', '9',
                                 '0', '='}):
            words_list.append(x[0].lower())
    return words_list
