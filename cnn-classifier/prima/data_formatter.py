import csv
import os.path
import codecs
import re
import config


def get_data(train_data_path, extension):
    label_list = []
    sent_list = []

    if extension == '.csv':
        with open(train_data_path, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='"')
            for row in reader:
                if not (row[0] or row[1] or row[2]):
                    continue
                label_list.append(int(row[0])-1)
                sent_list.append(' '.join([row[1], row[2]]))

    if extension == '.txt':
        pattern = _get_pattern(config.categories_num)
        with codecs.open(train_data_path, encoding='utf-8', mode='r') as f:
            for line in f:
                label = re.findall(pattern, line)
                sentence = re.split(pattern, line)[0]
                if not label:
                    continue
                label = label[0].strip('\t\n')
                label = label.split()
                label = [_set_label(x) for x in label]
                sent_list.append(sentence)
                label_list.append(label)
    return sent_list, label_list


def _set_label(x):
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
