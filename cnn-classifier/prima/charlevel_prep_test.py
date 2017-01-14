import sys
import os
import glob
import numpy as np

from subprocess import Popen, PIPE


def prepare_data(data_path, num_files, num_categories):
    folder = './split'
    if os.path.isdir(folder):
        for the_file in os.listdir(folder):
            file_path = os.path.join(folder, the_file)
            if os.path.isfile(file_path):
                os.unlink(file_path)

    process = Popen('./splitter.sh %s %s' % (data_path, num_files), shell=True)
    process.wait()

    files = os.listdir('./split')
    extension = os.path.splitext(data_path)[1]
    processes = []
    print('First process results:')
    devnull = open('/dev/null', 'w')
    file_path = '/'.join(['./split', files[0]])
    processes.append(Popen(['python', 'prep_run_test.py',
                            file_path, '001',
                            num_categories, extension]))
    for file in files[1:]:
        file_path = '/'.join(['./split', file])
        p = Popen(['python', 'prep_run_test.py',
                   file_path, str(files.index(file)+1).zfill(3),
                   num_categories, extension], stdout=devnull)
        processes.append(p)
    for p in processes:
        p.wait()
    print('All processes successfully finished!')

    files = glob.glob('./split/*.npz')
    X = np.empty((0, 1, 200, 350), np.int8)
    y = np.empty((0, int(num_categories)), np.int8)
    for file in files:
        npzfile = np.load(file)
        X = np.append(X, npzfile['X'], axis=0)
        y = np.append(y, npzfile['y'], axis=0)
    file_name = os.path.split(data_path)[1]
    file_path = '/'.join(['./tensor', file_name.split('.')[0]])
    np.savez(file_path, X=X, y=y)

if __name__ == '__main__':
    path = str(sys.argv[1])
    _num_files = str(sys.argv[2])
    _num_categories = str(sys.argv[3])
    prepare_data(path, _num_files, _num_categories)
