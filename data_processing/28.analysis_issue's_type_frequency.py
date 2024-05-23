from datetime import datetime
import os
import random
import numpy as np
import pandas as pd


def read_files(rank_path):
    f = open(rank_path)
    f.readline()
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

    return np.array(x_obj)

def read_enttiy2obj(entity_obj_path):
    X = pd.read_csv(entity_obj_path, sep="\t", header=None, dtype=str)

    return np.array(X)


def write_triples_2_id(path, data):
    try:
        fobj = open(path, 'w')

    except IOError as err:
        print('file open error: {0}'.format(err))

    else:
        num = len(data)
        fobj.writelines('%s\n' % num)
        for k in range(len(data)):
            _str = str(data[k][0]) + ' ' + str(data[k][1]) + ' ' + str(data[k][2]) + '\n'
            fobj.writelines('%s' % _str)

        fobj.close()
    print('WRITE FILE DONE!')


def write_rank(path, data):
    try:
        fobj = open(path, 'w')

    except IOError as err:
        print('file open error: {0}'.format(err))

    else:
        num = len(data)
        fobj.writelines('%s\n' % num)
        for k in range(len(data)):
            _str = str(data[k][0]) + '-->' + str(data[k][1]) + '-->' + str(data[k][2]) + '\n'
            fobj.writelines('%s' % _str)

        fobj.close()
    print('WRITE FILE DONE!')


def write_triples_with_created_date(out_path, data):
    ls = os.linesep
    num = len(data)

    try:
        fobj = open(out_path, 'w')
    except IOError as err:
        print('file open error: {0}'.format(err))

    else:
        fobj.writelines('%s\n' % num)
        for j in range(num):
            #
            _str = data[j][0] + '\t' + data[j][1] + '\t' + data[j][2] + '\t' + data[j][3] + '\n'

            fobj.writelines('%s' % _str)

        fobj.close()

    print('WRITE FILE DONE!')


def _read_tsv(input_file):
    """Reads a tab separated value file.

    修改：
    reading train.csv will be changed with reading train2id.txt, obtain index
    """
    data = pd.read_csv(input_file)  #
    data = np.array(data)
    data_id = []
    for i in range(len(data)):
        _tmp = data[i][0]
        tmp = _tmp.split(' ')
        if tmp:
            id_list = []
            for s in tmp:
                id_list.append(s.strip())
            data_id.append(id_list)

    return np.array(data_id)


def write_data_2_id(path, data):
    try:
        fobj = open(path, 'w')

    except IOError as err:
        print('file open error: {0}'.format(err))

    else:
        num = len(data)

        fobj.writelines('%s\n' % num)
        for k in range(num):
            _str = str(data[k][0]) + '\t' + str(data[k][1]) + '\n'
            fobj.writelines('%s' % _str)

        fobj.close()
    print('WRITE FILE DONE!')


def read_rank_result(file_path):
    f = open(file_path)
    f.readline()
    x_obj = []
    for d in f:
        d = d.strip()
        elements = []
        if d:
            d = d.split('-->')

            elements.append(d[0])
            elements.append(int(d[1]))
            elements.append(int(d[2]))
            x_obj.append(elements)
    f.close()

    return x_obj

def read_rank_result_hadoop77K(file_path):

    f = open(file_path)
    x_obj = []
    for d in f:
        d = d.strip()
        elements = []
        if d:
            d = d.split('\t')
            _tc = d[0].split("-->")
            elements.append(_tc[0])
            elements.append(int(_tc[1]))
            elements.append(int(d[-1]))
            x_obj.append(elements)
    f.close()

    return x_obj

def compare_two_time(first_time,second_time):
    # print(first_time,second_time)
    # 日期格式话模版
    format_pattern = "%Y-%m-%d %H:%M:%S"

    # first_time = "2022-09-06 13:51:32"
    # second_time = datetime.now()
    # print(end_date) # datetime.datetime(2018, 10, 15, 11, 19, 52, 186250)
    # print(type(end_date)) # <type 'datetime.datetime'>

    # 将 'datetime.datetime' 类型时间通过格式化模式转换为 'str' 时间
    # second_time = second_time.strftime(format_pattern)
    # print(second_time, type(second_time)) # ('2018-10-15 11:21:44', <type 'str'>)

    # 将 'str' 时间通过格式化模式转化为 'datetime.datetime' 时间戳, 然后在进行比较
    _first_time = datetime.strptime(first_time, format_pattern)

    _second_time = datetime.strptime(second_time, format_pattern)

    if _first_time < _second_time:
        return True


def static_frequency(dic_path, data):
    d = data
    c = dict.fromkeys(d, 0)
    for x in d:
        c[x] += 1
    sorted_x = sorted(c.items(), key=lambda d: d[1], reverse=True)
    total_issues = len(data)
    num = len(sorted_x)
    file = open(dic_path, 'w')
    file.writelines('%s\n' % num)
    i = 0
    for e in sorted_x:
        file.write(str(i) + '\t' + str(e[0]) + '\t' + str(e[1]) + '\t' + str(float(e[1]/total_issues)) + '\n')
        i += 1

    file.close()

if __name__ == "__main__":

    entity2obj = read_enttiy2obj("../data/RedHat/ID_Name_Project_Type_Status_sMention_Time.txt")
    row, colum = entity2obj.shape
    print(row, colum)
    entity_type = []
    for i in range(row):
        entity_type.append(entity2obj[i][4])

    static_frequency("../data/RedHat/issue_type_frequency.txt",entity_type)



