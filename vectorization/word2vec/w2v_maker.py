import os
import codecs
import nltk.tokenize

from gensim.models import Word2Vec


_tokenizer = nltk.tokenize.RegexpTokenizer(pattern=r'[\w\$]+|[^\w\s]')
params = {
    "save_path": './w2v/w2v_model.bin',
    "vect_size": 256,
    "min_count": 1,
    "vocab_max_size": 10000,
    "win_size": 5,
    "workers_num": 25
}


def prepare_data(file_path):
    input_file = codecs.open(file_path, encoding='utf-8', mode='r', errors='ignore')
    result = []
    for line in input_file:
        tokens = _tokenizer.tokenize(line.lower())
        result.append(tokens)
    return result


def w2v_train(tokenized_lines):
    dir_remove('./w2v')
    model = Word2Vec(window=int(params['win_size']),
                     min_count=int(params['min_count']),
                     max_vocab_size=int(params['vocab_max_size']),
                     size=int(params['vect_size']),
                     workers=int(params['workers_num']))
    model.build_vocab(tokenized_lines)
    model.train(tokenized_lines)
    model.save(params['save_path'])
    return model


def dir_remove(path):
    if (os.path.exists(path)) and (os.listdir(path) != []):
        shutil.rmtree(path)
        os.makedirs(path)

    if not os.path.exists(path):
        os.makedirs(path)
