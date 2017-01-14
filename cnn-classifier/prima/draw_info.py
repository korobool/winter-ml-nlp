import yaml
import matplotlib.pyplot as plt
import numpy as np


def draw_acc():
    file_list = yaml.load(open('./logs/list.yaml', 'r'))
    all_data_list = []
    for file in file_list:
        data_dict = yaml.load(open(file, 'r'))
        all_data_list.extend(data_dict['acc'])
    plt.figure(figsize=(20, 10))
    plt.plot(all_data_list)


def draw_loss():
    file_list = yaml.load(open('./logs/list.yaml', 'r'))
    all_data_list = []
    for file in file_list:
        data_dict = yaml.load(open(file, 'r'))
        all_data_list.extend(data_dict['loss'])
    plt.figure(figsize=(20, 10))
    plt.plot(all_data_list)
    return all_data_list[0]


def draw_val_acc(**kwargs):
    val_acc_start = kwargs['val_acc_start']
    # print(val_acc_start)
    file_list = yaml.load(open('./logs/list.yaml', 'r'))
    all_data_list = []
    for file in file_list:
        data_dict = yaml.load(open(file, 'r'))
        all_data_list.append(data_dict['val_acc'])
    plt.figure(figsize=(20, 10))
    all_data_list.insert(0, val_acc_start)
    plt.plot(all_data_list)


def draw_val_loss(**kwargs):
    val_loss_start = kwargs['val_loss_start']
    # print(val_loss_start)
    file_list = yaml.load(open('./logs/list.yaml', 'r'))
    all_data_list = []
    for file in file_list:
        data_dict = yaml.load(open(file, 'r'))
        all_data_list.append(data_dict['val_loss'])
    plt.figure(figsize=(20, 10))
    all_data_list.insert(0, val_loss_start)
    plt.plot(all_data_list)