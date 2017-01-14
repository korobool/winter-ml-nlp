import sys
import os
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
    processes.append(Popen(['python', 'prep_run_train.py',
                            file_path, '001',
                            num_categories, extension]))
    for file in files[1:]:
        file_path = '/'.join(['./split', file])
        p = Popen(['python', 'prep_run_train.py',
                   file_path, str(files.index(file)+1).zfill(3),
                   num_categories, extension], stdout=devnull)
        processes.append(p)
    for p in processes:
        p.wait()
    print('All processes successfully finished!')


if __name__ == '__main__':
    path = str(sys.argv[1])
    _num_files = str(sys.argv[2])
    _num_categories = str(sys.argv[3])
    prepare_data(path, _num_files, _num_categories)
