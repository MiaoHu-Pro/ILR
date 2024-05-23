

import math
import os
import pickle
from functools import partial
from multiprocessing import Pool
import re

import numpy as np
import random

import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm


def test():
    s = np.random.randint(low=0, high=10, size=5)
    list1 = [i for i in range(10)]
    num1 = random.sample(list1, 8)
    print(num1)


def read_entity2obj(entity_obj_path):
    """
    14344(index) 	/m/0wsr(symbol) 	 Atlanta Falcons(label)	 American football team (description)
    :param entity_obj_path:
    :return:
    """
    f = open(entity_obj_path)

    x_obj = []
    for d in f:
        d = d.strip()
        if d:
            d = d.split('\t')

            elements = []
            for n in d:
                elements.append(n.strip())
            d = elements

            x_obj.append(d)

    f.close()
    X = np.array(x_obj)

    return X


def write_to_file_entity_obj(out_path, all_data):
    ls = os.linesep
    leng = len(all_data)
    try:
        fobj = open(out_path, 'w')
    except IOError as err:
        print('file open error: {0}'.format(err))

    else:
        for j in range(leng):
            #
            _str = str(j) + '\t' + all_data[j][0] + '\t' + all_data[j][1] + '\t' + all_data[j][2] + '\n'

            fobj.writelines('%s' % _str)

        fobj.close()

    print('WRITE FILE DONE!')

def file_name(file_dir):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.pkl':
                L.append(file)
    return L

def _read_tsv(input_file):
    """Reads a tab separated value file.

    修改：
    reading train.csv will be changed with reading train2id_1_5.txt, obtain index
    """
    data = pd.read_csv(input_file)  #
    data = np.array(data)
    data_id = []
    for i in range(len(data)):
        _tmp = data[i][0]
        tmp = _tmp.split('\t')
        if tmp:
            id_list = []
            for s in tmp:
                id_list.append(s.strip())
            data_id.append(id_list)

    return data_id

def _read_triple_tsv(input_file):
    """Reads a tab separated value file.

    修改：
    reading train.csv will be changed with reading train2id_1_5.txt, obtain index
    """
    data = pd.read_csv(input_file)  #
    data = np.array(data)
    data_id = []
    for i in range(len(data)):
        _tmp = data[i][0]
        tmp = _tmp.split('\t')
        if tmp:
            id_list = []
            for s in tmp:
                id_list.append(s.strip())
            data_id.append(id_list)

    return data_id

def write_triples_2_id(path, data):
    try:
        fobj = open(path, 'w')

    except IOError as err:
        print('file open error: {0}'.format(err))

    else:
        num = len(data)
        fobj.writelines('%s\n' % num)
        for k in range(len(data)):
            # _str = str(data[k][0]) + '\t' + str(data[k][1]) + '\n'
            _str = str(data[k][0]) + '\t' + str(data[k][1]) + '\t' + str(data[k][2]) + '\n'
            fobj.writelines('%s' % _str)

        fobj.close()
    print('WRITE FILE DONE!')

def combine_m_f_r():
    m_f_path = "./data/WN18RR/test2id_m_f.txt"
    m_r_path = "./data/WN18RR/test2id_m_r.txt"

    m_f = _read_tsv(m_f_path)
    m_r = _read_tsv(m_r_path)

    m_f_r = m_f + m_r

    print(len(m_f))
    print(len(m_r))

    print(len(m_f_r))

    write_triples_2_id("./data/WN18RR/test2id_m_f_r.txt",m_f_r)


def read_obj(entity_obj_path):

    X = pd.read_csv(entity_obj_path,sep="\t",header=None)

    return np.array(X)


def test2id_split(dir_path, test_triple_path):

    test2id = _read_tsv(test_triple_path)
    n = 1
    sub_test_set = []
    for i in range(len(test2id)):

        sub_test_set.append(test2id[i])
        if len(sub_test_set) == 1000:
            write_triples_2_id(dir_path+"test_set_new2new2id_3723_" + str(n) + ".txt", sub_test_set)
            sub_test_set = []
            n += 1

    write_triples_2_id(dir_path+"test_set_new2new2id_3723_" + str(n) + ".txt", sub_test_set)


def random_select_test_samples(dir_path, test_triple_path, random_num):

    test2id = _read_tsv(test_triple_path)
    total_test_triple = len(test2id)
    total_index = [i for i in range(total_test_triple)]

    from random import sample
    selection_index = sample(total_index, random_num)

    sub_test_set = []
    for i in selection_index:
        sub_test_set.append(test2id[i])
    if len(sub_test_set) == random_num:
        write_triples_2_id(dir_path+"test2id_r"+str(random_num)+".txt", sub_test_set)
        print("DONE")





if __name__ == "__main__":

    data_file = ['Apache', 'Jira', 'Mojang', 'MongoDB', 'Qt', 'RedHat']


    for _name_file in data_file:

        dir_path = "../data/" + _name_file + "/"
        test2id_path = "../data/" + _name_file + "/test2id.txt"

        # test2id_split(dir_path,test2id_path)

        random_select_test_samples(dir_path,test2id_path, random_num=1000)





