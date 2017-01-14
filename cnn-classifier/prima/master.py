import os
import shutil
import yaml
import getopt
import sys
import config

from subprocess import Popen


def master():
    force = False
    myopts, args = getopt.getopt(sys.argv[1:], 'f', ['force'])
    for o, a in myopts:
        if o in ('-f', '--force'):
            force = True

    # process = Popen('python data_loader.py', shell=True)
    # process.wait()
    if (not os.path.exists('./tensor/train')) or \
            (not os.path.isfile('./info.yaml')) or\
            (os.listdir('./tensor/train') == []) or force:
        process = Popen('python data_maker.py', shell=True)
        process.wait()

        dir_remove('./tensor/train')

        print('Making tensor from train file...')
        process = Popen('python charlevel_prep_train.py train.txt 12 ' + str(config.categories_num), shell=True)
        process.wait()

        print('Making tensor from test file...')
        process = Popen('python charlevel_prep_test.py test.txt 12 ' + str(config.categories_num), shell=True)
        process.wait()

    stream = open('info.yaml', 'r')
    info = yaml.load(stream)

    model_path = './model'
    dir_remove(model_path)
    dir_remove('./logs')
    log_path = './log.txt'
    train_npz_path = './tensor/train'
    test_npz_path = './tensor/test.npz'
    process = Popen('python -u charlevel_cnn_run.py %s %s %s %s | tee %s' % (train_npz_path,
                                                                             test_npz_path,
                                                                             model_path,
                                                                             info['train_shape'],
                                                                             log_path), shell=True)
    process.wait()


def dir_remove(path):
    if (os.path.exists(path)) and (os.listdir(path) != []):
        shutil.rmtree(path)
        os.makedirs(path)

    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == '__main__':
    master()