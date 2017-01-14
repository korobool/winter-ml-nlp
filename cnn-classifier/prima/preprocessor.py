import codecs
import re
import numpy as np
import random
import os
import sys


def prepare_data(categories_data_path, categories, prefix):
    n_label = len(categories)
    pattern = _get_pattern(n_label)

    sent_learn = file_reader(categories_data_path, pattern)
    zero_counts = []
    one_counts = []
    for category in categories:
        label_one = [x[1][categories.index(category)] for x in sent_learn]
        one_ = 0
        zero_ = 0
        for i in label_one:
            if i == 1:
                one_ += 1
            else:
                zero_ += 1
        zero_counts.append(zero_)
        one_counts.append(one_)
    for category in categories:
        print('Category %s zeroes=%s, ones=%s' % (category,
                                                  zero_counts[categories.index(category)],
                                                  one_counts[categories.index(category)]))

    for category in categories:
        one_list = []
        zero_list = []
        i = categories.index(category)

        for x in sent_learn:
            if x[1][i] == 1:
                one_list.append((x[0], 1))
            else:
                zero_list.append((x[0], 0))

        if one_counts[i] < zero_counts[i]:
            choices_one = one_list
            choices_zero = zero_list
            # choices_zero = random.sample(zero_list, one_counts[i])
        else:
            choices_one = one_list
            choices_zero = zero_list

        balanced_data = sorted(choices_one, key=lambda k: random.random())
        file_path = '/'.join(['./ctgrs', prefix+category+'.txt'])
        with codecs.open(file_path, encoding='utf-8', mode='w') as fp:
            for x in balanced_data:
                fp.write("%s\n" % x[0])


def zero_one_count(dir_):
    files = os.listdir(dir_)
    pattern = _get_pattern(1)
    for file in files:
        one_ = 0
        zero_ = 0
        sent_list = file_reader('/'.join([dir_, file]), pattern)
        for sent in sent_list:
            if int(sent[1][0]) == 1:
                one_ += 1
            else:
                zero_ += 1
        print('For file %s zeroes=%s, ones=%s' % (file, zero_, one_))


def _set_label(x):
    if int(x) >= 1:
        label_ = 1
    else:
        label_ = 0
    return label_


def _get_pattern(n):
    if n == 1:
        return '[\t\b][0-9][\n]'
    else:
        return '[\t\b]' + '[0-9][\t]' * (n - 1) + '[0-9][\n]'


def file_reader(path, pattern):
    sent_learn = []
    sent_list = []
    with codecs.open(path, encoding='utf-8',
                     mode='r') as f:
        for line in f:
            label = re.findall(pattern, line)
            sentence = re.split(pattern, line)[0]
            if not label:
                continue
            label = label[0].strip('\t\n')
            label = label.split()
            label = [_set_label(x) for x in label]
            sent_learn.append((sentence, label))
            sent_list.append(sentence)
    print('There is %s raw items.' % len(sent_learn))
    sent_list = list(set(sent_list))
    print('There is %s unique sentences.' % len(sent_list))
    sent_learn_clear = []
    for sent in sent_list:
        label_one = []
        sys.stdout.write("\rSentence number %s." % str(sent_list.index(sent) + 1))
        sys.stdout.flush()
        for tuple_one in sent_learn:
            if sent == tuple_one[0]:
                if not label_one:
                    label_one = tuple_one[1]
                else:
                    label_cmp = []
                    for i, j in zip(label_one, tuple_one[1]):
                        if i == 1 or j == 1:
                            label_cmp.append(1)
                        else:
                            label_cmp.append(0)
                    label_one = label_cmp
        sent_learn_clear.append((sent, label_one))
    print('\n')
    return sent_learn_clear