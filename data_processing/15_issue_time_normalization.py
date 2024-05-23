

import math
import os
import numpy as np
from datetime import datetime

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
            _str = data[j][0] + '\t' + data[j][1] + '\n'

            fobj.writelines('%s' % _str)

        fobj.close()

    print('WRITE FILE DONE!')


def write_triples_to_csv(out_path, data):
    num = len(data)

    try:
        fobj = open(out_path, 'w')
    except IOError as err:
        print('file open error: {0}'.format(err))

    else:
        fobj.writelines('Issues,Relationship,Link Issues,time\n')
        for j in range(num):
            #
            _str = data[j][0] + ',' + data[j][1] + ',' + data[j][2] + ',' + data[j][3] + '\n'

            fobj.writelines('%s' % _str)

        fobj.close()

    print('WRITE FILE DONE!')


def gap_time(head_with_time, tail_with_time):
    format_pattern = "%Y-%m-%d %H:%M:%S"
    head_with_time = datetime.strptime(head_with_time, format_pattern)
    tail_with_time = datetime.strptime(tail_with_time, format_pattern)
    gap = math.fabs((head_with_time - tail_with_time).total_seconds())
    gap_hour = round(gap / 3600, 3)
    # logger.info("gap_hour:{0}".format(gap_hour))
    return gap_hour


def write_data2id(path, data):
    try:
        fobj = open(path, 'w')

    except IOError as err:
        print('file open error: {0}'.format(err))

    else:
        num = len(data)

        fobj.writelines('%s\n' % num)
        for key, value in data.items():
            _str = key + "\t" + str(value) + '\n'
            fobj.writelines('%s' % _str)

        fobj.close()
    print('WRITE FILE DONE!')


def write_ID_Name_Mention_Time(out_path, data):
    ls = os.linesep
    num = len(data)

    try:
        fobj = open(out_path, 'w')
    except IOError as err:
        print('file open error: {0}'.format(err))

    else:
        num = len(data)

        fobj.writelines('%s\n' % num)
        for j in range(num):
            #
            _str = data[j][0] + '\t' + data[j][1] + '\t' + data[j][2] + '\t' + data[j][3] + '\t' + str(
                data[j][4]) + '\n'

            fobj.writelines('%s' % _str)

        fobj.close()

if __name__ == "__main__":

    # hadoop_77K : 2006.2.1 ~ 2022.9.12
    # MongoDB : 2009-07-30 17:20:29 ~ 2022-10-07 19:34:10

    entity2id2time = read_files("../data/Qt/Qt_entity_id_time.txt")
    years_frequency = {}

    for i in range(len(entity2id2time)):

        if entity2id2time[i][-1].split("-")[0] in years_frequency:
            years_frequency[entity2id2time[i][-1].split("-")[0]] += 1
        else:
            years_frequency[entity2id2time[i][-1].split("-")[0]] = 1
    print(years_frequency)


    values = list(years_frequency.values())
    print(np.mean(values[1:-1]))


# apache
# {'2022': 711, '2021': 67186, '2020': 67115, '2019': 68542, '2018': 77981, '2017': 86883, '2016': 94680, '2015': 92455, '2014': 78159, '2013': 58065, '2012': 47899, '2011': 43544, '2010': 38980, '2009': 41596, '2008': 36617, '2004': 8930, '2003': 6772, '2006': 32530, '2005': 22561, '2007': 36387, '2002': 5212, '2001': 2115, '2000': 6}
# 48295.666666666664


# Mojang
# {'2022': 203, '2021': 80237, '2020': 110307, '2019': 54692, '2017': 25759, '2018': 30618, '2015': 23073, '2016': 24108, '2014': 30625, '2013': 35255, '2012': 5942}
# 46074.88888888889 move first and last year

# MongoDB
# {'2022': 52, '2021': 20170, '2020': 18483, '2019': 15663, '2018': 16174, '2017': 10912, '2016': 12232, '2015': 11955, '2014': 9897, '2013': 7536, '2012': 6431, '2010': 2797, '2009': 795, '2011': 4075}
# 11087.083333333334
