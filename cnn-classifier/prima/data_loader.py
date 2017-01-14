import os
import time
import shutil

from ftplib import FTP


def copy_ftp_files(remote_path, local_path):
    ftp = FTP()
    ftp.connect('hq.ai-labs.org', 8021)
    ftp.login('kylabs', 'gl0bus')
    ftp.cwd(remote_path)

    print("Copying...")
    transfer_list = ftp.nlst()
    files_copied = 0

    for fl in transfer_list:
        # create a full local filepath
        local_file = '/'.join([local_path, fl])
        print(local_file)
        #open a the local file
        fileObj = open(local_file, 'wb')
        # Download the file a chunk at a time using RETR
        ftp.retrbinary('RETR ' + fl, fileObj.write)
        # Close the file
        fileObj.close()
        files_copied += 1

    print('Copied %s files on %s' % (str(files_copied), time_stamp()))
    ftp.close() # Close FTP connection


def time_stamp():
    return str(time.strftime("%a %d %b %Y %I:%M:%S %p"))


def dir_remove(path):
    if (os.path.exists(path)) and (os.listdir(path) != []):
        shutil.rmtree(path)
        os.makedirs(path)

    if not os.path.exists(path):
        os.makedirs(path)

if __name__ == '__main__':
    for dir_name in ['ctgrs', 'ctgrs_negative']:
        remote_dir_path = '/'.join(['repnup/user_labled', dir_name])
        local_dir_path = '/'.join(['.', dir_name])
        dir_remove(local_dir_path)
        copy_ftp_files(remote_dir_path,local_dir_path)
