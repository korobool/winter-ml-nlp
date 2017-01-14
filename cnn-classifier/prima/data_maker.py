import os
import codecs
import math
import shutil
import random
import yaml

from pathlib import Path
from collections import defaultdict


class PosCatNotMatchesToNegCat(Exception):
    pass


def make_txt(pos_path, neg_path, train_percent):
    pos_list = os.listdir(pos_path)
    # neg_list = os.listdir(neg_path)
    categories = [str(x.split('.')[0]) for x in pos_list]
    n_categories = len(categories)

    # for x in categories:
    #    if x+'_negative.txt' not in neg_list:
    #        print(x+'_negative.txt')
    #        raise PosCatNotMatchesToNegCat('Positive categories meaning'
    #                                       ' not matches to negative.')

    fp_train_path = './train.txt'
    fp_test_path = './test.txt'
    fp_train = codecs.open(fp_train_path,
                           encoding='utf-8', mode='w')
    fp_test = codecs.open(fp_test_path,
                          encoding='utf-8', mode='w')


    pos_all_dict = defaultdict(list)
    for x in categories:
        file_name = x + '.txt'
        print(file_name)
        with codecs.open(str(Path(pos_path) / file_name),
                         encoding='utf-8', mode='r') as fp:
            for line in fp:
                line = line.strip('\n')
                pos_all_dict[line].append(categories.index(x))
                # if len(pos_all_dict[line]) > 1:
                #     print(line, pos_all_dict[line])

    main_list = []
    for item in pos_all_dict.items():
        label = mask_decode(item[1], n_categories)
        main_list.append(item[0] + label + '\n')

    train_pos_count = math.trunc(len(main_list)*(1-train_percent))
    main_train = main_list[:train_pos_count]
    main_test = main_list[train_pos_count:]

    main_train = sorted(main_train, key=lambda k: random.random())
    # main_train = main_train[:int(len(main_train)/40)]

    main_test = sorted(main_test, key=lambda k: random.random())
    # main_test = main_test[:int(len(main_test)/40)]

    for item in main_train:
        fp_train.write(item)

    for item in main_test:
        fp_test.write(item)

    file_path = './categories.yaml'
    with open(file_path, 'w') as info_yaml:
        info_yaml.write(yaml.dump(categories,
                                  default_flow_style=False,
                                  allow_unicode=True))

    file_path = './info.yaml'
    info = dict(train_shape=len(main_train),
                test_shape=len(main_test))
    with open(file_path, 'w') as info_yaml:
        info_yaml.write(yaml.dump(info,
                                  default_flow_style=False,
                                  allow_unicode=True))



def mask_decode(label, n):
    mask = [0]*n
    label_set = set(label)
    label_list = list(label_set)
    for i in label_list:
        mask[int(i)] = 1
    str_label = ''
    for j in mask:
        str_label = str_label + '\t' + str(j)
    return str_label



if __name__ == '__main__':
    pos_path_ = './ctgrs'
    neg_path_ = './ctgrs_negative'
    train_percent_ = 0.2
    print('Data maker running...')
    make_txt(pos_path_, neg_path_, train_percent_)
