

import os
from datetime import time
from datetime import datetime
import numpy as np
import pandas as pd

num = 0

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


def read_entity2obj(entity_obj_path):
    """
    14344(index) 	/m/0wsr(symbol) 	 Atlanta Falcons(label)	 American football team (description)
    :param entity_obj_path:
    :return:
    """
    # f = open(entity_obj_path)
    #
    # x_obj = []
    # for d in f:
    #     d = d.strip()
    #     if d:
    #         d = d.split('\t')
    #
    #         elements = []
    #         for n in d:
    #             elements.append(n.strip())
    #         d = elements
    #         x_obj.append(d)
    #
    # f.close()
    # X = np.array(x_obj)
    # return X

    X = pd.read_csv(entity_obj_path, sep="\t", header=None,dtype=str)

    return np.array(X)


def compare_two_time(first_time, second_time):
    # print(first_time, second_time)
    # 日期格式话模版
    format_pattern = "%Y-%m-%d %H:%M:%S"

    global num

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

    if _first_time > _second_time:
        num +=1
        l_time =  str(_first_time)
    else:
        l_time =  str(_second_time)

    if _first_time > _second_time:
        latest_time = _first_time
        previous_time = _second_time
    else:
        latest_time = _second_time
        previous_time = _first_time

    gap = (latest_time - previous_time).total_seconds()

    gap_hour = round(gap/(24*3600),3)
    # print(l_time, gap_hour)
    return l_time, gap_hour


def write_triples_with_created_date(out_path,data):

    ls = os.linesep
    num = len(data)

    try:
        fobj = open(out_path,  'w')
    except IOError as err:
        print('file open error: {0}'.format(err))

    else:
        fobj.writelines('%s\n' % num)
        for j in range(num):
            #
            _str = data[j][0] + '\t' + data[j][1] + '\t' + data[j][2]+ '\t' + data[j][3] + '\t' + data[j][4] + '\t' + data[j][5]+ '\t' + data[j][6] + '\n'

            fobj.writelines('%s' % _str)

        fobj.close()

    print('WRITE FILE DONE!')

if __name__ == "__main__":



    data_file = ['Apache', 'Jira', 'Mojang', 'MongoDB', 'Qt', 'RedHat']


    for _name_file in data_file:
        print(_name_file)
        train_path = "../data/" + _name_file + "/test.txt"
        total_triples = read_files(train_path)
        row, colum = total_triples.shape
        entity_name_createdtime = read_entity2obj("../data/" + _name_file + "/ID_Name_Project_Type_Status_sMention_Time.txt")

        entity_date = entity_name_createdtime[:, -1].tolist()
        entity_name = entity_name_createdtime[:, 1].tolist()

        entity_2_date = {}
        for i in range(len(entity_name)):
            entity_2_date[entity_name[i]] = entity_date[i]

        total_triples_with_time = []
        #
        gap_time = []
        for i in range(row):
            # print(i)
            _triple = []

            head = total_triples[i][0]
            rel = total_triples[i][1]
            tail = total_triples[i][2]

            head_time = entity_2_date.get(str(head).strip())
            tail_time = entity_2_date.get(str(tail).strip())

            later_time,_gap_time = compare_two_time(head_time,tail_time)
            gap_time.append(_gap_time)
            # _gap_time = gap_time(head_time,tail_time)
            _triple.append(head)
            _triple.append(rel)
            _triple.append(tail)
            _triple.append(head_time)
            _triple.append(tail_time)
            _triple.append(later_time)
            _triple.append(str(_gap_time))

            total_triples_with_time.append(_triple)

        # write_triples_with_created_date("../data/" + _name_file + "/gap_time_analysis_for_training.txt",total_triples_with_time)

        gap_time_nor_with_seven_day = []
        for i in range(len(gap_time)):
            gap_time_nor_with_seven_day.append(round(gap_time[i]/365)) # 365days

        d = gap_time_nor_with_seven_day
        c = dict.fromkeys(d, 0)
        for x in d:
            c[x] += 1
        sorted_x = sorted(c.items(), key=lambda d: d[1], reverse=True)

        print("sorted_x:", sorted_x[:10])

        print(sorted_x[0][1]/row, "\n")

    # print("min(gap_time):",min(gap_time))
    # print("max(gap_time):",max(gap_time))
    # print("mean(gap_time):",np.mean(gap_time))
    #
    # print("head time later than tail global: ",num,num/row)

"""

30% -- 40% of triples are constructed within 336 hours (2 weeks)

40%-50% of triples are constructed within 8 weeks

HADOOP-234	depends upon	HADOOP-1251	2007-04-12 05:54:52

HADOOP-234  2006-05-19 16:28:42

HADOOP-1251 : 2007-04-12 05:54:52


"""





