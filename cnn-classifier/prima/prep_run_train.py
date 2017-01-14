import numpy as np
import sys
import csv
import codecs
import config

from data_formatter import get_data


def get_train_data(train_data_path, file_name, num_categories, file_type):
    chars = config.characters
    batch = config.batch

    indexes = list(range(len(chars)))
    couples = dict(zip(chars, indexes))

    print("Getting sequences...")

    X = np.empty((0, 1, len(chars), 350), np.int8)
    y = []
    sent_num = 0

    sentence_list, label_list = get_data(train_data_path, file_type)
    max_num = int(num_categories)
    file_num = 1

    for sentence_raw in sentence_list:
        sent_num += 1
        frame = np.zeros((len(chars), 350), dtype=np.int8)
        sentence = sentence_raw.lower()
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
        y.append(label_list[sentence_list.index(sentence_raw)])
        sys.stdout.write("\rSentence number %s. Shape of X: %s. Current batch: %s."
                         % (sent_num, X.shape, file_num))
        sys.stdout.flush()
        if len(y) == batch:
            file_path = 'train_process_' + file_name + '_batch_' + str(file_num).zfill(5)
            np.savez('/'.join(['./tensor/train', file_path]), X=X, y=np.asarray(y))
            file_num += 1
            y = []
            X = np.empty((0, 1, len(chars), 350), np.int8)

    file_path = 'train_process_' + file_name + '_batch_' + str(file_num).zfill(5)
    np.savez('/'.join(['./tensor/train', file_path]), X=X, y=np.asarray(y))
    print("\nAll tensor batches for train data were successfully saved...")

if __name__ == '__main__':
    path = str(sys.argv[1])
    file = str(sys.argv[2])
    _num_categories = str(sys.argv[3])
    _file_type = str(sys.argv[4])
    get_train_data(path, file, _num_categories, _file_type)
