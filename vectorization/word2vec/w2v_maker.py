# Make Word2Vec model.

import os
import codecs
import shutil
import nltk.tokenize

from gensim.models import Word2Vec


# A tokenizer that splits a string using a regular expression, which
# matches either the tokens or the separators between tokens.

_tokenizer = nltk.tokenize.RegexpTokenizer(pattern=r'[\w\$]+|[^\w\s]')

# Parameters of Word2Vec model:
# - save_path - the path for model saving;
# - vect_size - the dimensionality of the feature vectors
# - min_count - ignore all words with total frequency lower than this;
# - max_vocab_size - limit RAM during vocabulary building; if there are more unique words than this,
#   then prune the infrequent ones. Every 10 million word types need about 1GB of RAM.
#   Set to None for no limit (default).
# - window - the maximum distance between the current and predicted word within a sentence
# - workers_num - use this many worker threads to train the model
params = {
    "save_path": './w2v/w2v_model.bin',
    "vect_size": 256,
    "min_count": 1,
    "max_vocab_size": 100000,
    "window": 5,
    "workers_num": 4
}


def prepare_data(file_path):
    """
    Prepare data for making Word2vec model
    :param file_path: path to the file with data for making model
    :return: tokenized sentences
    """
    input_file = codecs.open(file_path, encoding='utf-8', mode='r', errors='ignore')
    result = []
    for line in input_file:
        tokens = _tokenizer.tokenize(line.lower())
        result.append(tokens)
    return result


def w2v_train(tokenized_lines):
    """
    Make the Word2Vec model on the base of tokenized sentences
    :param tokenized_lines: tokenized sentences
    :return: Word2Vec model
    """
    dir_remove('./w2v')
    model = Word2Vec(window=int(params['window']),
                     min_count=int(params['min_count']),
                     max_vocab_size=int(params['max_vocab_size']),
                     size=int(params['vect_size']),
                     workers=int(params['workers_num']))
    model.build_vocab(tokenized_lines)
    model.train(tokenized_lines)
    model.save(params['save_path'])
    return model


def dir_remove(path):
    """
    Remove the dir with model with all subdirs if it exists.
    Make the dir for model if it is not exists.
    :param path: path to the model
    """
    if (os.path.exists(path)) and (os.listdir(path) != []):
        shutil.rmtree(path)
        os.makedirs(path)

    if not os.path.exists(path):
        os.makedirs(path)
