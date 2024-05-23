import os
from datetime import datetime

import numpy as np


def write_issue_issue_links(path, data):
    try:
        fobj = open(path, 'w')

    except IOError as err:
        print('file open error: {0}'.format(err))

    else:
        num = len(data)
        fobj.writelines('%s\n' % num)
        for k in range(len(data)):
            _str = str(data[k][0]) + '\t' + str(data[k][1]) + '\t' + str(data[k][2]) + '\t' + str(data[k][3]) + '\n'
            fobj.writelines('%s' % _str)

        fobj.close()
    print('WRITE FILE DONE!')


def read_issue_links_with_time(rank_path):
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


def file_name(file_dir):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.txt':
                L.append(file)
    return L


if __name__ == "__main__":

    # 90 percent for training, 10 percent for testing
    # split_time = {"Mojang": "2021-02-06 23:59:59",
    #               "MongoDB": "2020-10-08 23:59:59",
    #               "Qt": "2020-02-23 23:59:59",
    #               "Jira": "2018-01-26 23:59:59",
    #               "Apache": "2019-12-19 23:59:59",
    #               "RedHat": "2019-12-06 23:59:59"
    #               }


    # 80 percent for training, 20 percent for testing
    split_time = {"Mojang": "2019-12-06 23:59:59",
                  "MongoDB": "2019-06-08 23:59:59",
                  "Qt": "2018-04-23 23:59:59",
                  "Jira": "2014-03-26 23:59:59",
                  "Apache": "2017-10-10 23:59:59",
                  "RedHat": "2017-11-06 23:59:59"
                  }


    format_pattern = "%Y-%m-%d %H:%M:%S"



    file_name = ["Mojang", "MongoDB", "Qt", "Jira", "Apache", "RedHat"]
    for _folder in file_name:
        print(_folder)
        _triples = read_issue_links_with_time("./datasets/" + _folder + "/" + _folder + "_issue_links_unification.txt")

        _split_time = split_time[_folder]
        _train = []
        _test = []

        for i in range(len(_triples)):
            if datetime.strptime(_triples[i][3] , format_pattern)  <= datetime.strptime(_split_time , format_pattern):
                _train.append(_triples[i])
            else:
                _test.append(_triples[i])

        write_issue_issue_links("./datasets/" + _folder + "/train_issue_links_80_pre.txt", _train)
        write_issue_issue_links("./datasets/" + _folder + "/test_issue_links_20_pre.txt", _test)
