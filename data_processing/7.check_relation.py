
import os
from utilities import clean
import numpy as np


def write_relations(path, data):
    try:
        fobj = open(path, 'w')

    except IOError as err:
        print('file open error: {0}'.format(err))

    else:
        num = len(data)
        fobj.writelines('%s\n' % num)
        for k in range(len(data)):
            _str = str(data[k][0]) + '\t' + str(data[k][1]) + '\t' + str(data[k][2]) + '\n'
            fobj.writelines('%s' % _str)

        fobj.close()
    print('WRITE FILE DONE!')


def read_file(rank_path):
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


# file_name_list = file_name(issuesFilePath)

if __name__ == "__main__":

    relation = read_file('datasets/relation2id_original.txt')

    relation_clean = []
    for _relation in relation:

        words = clean(_relation[1])
        _relation_clean = []
        _relation_clean.append(_relation[0])
        _relation_clean.append(" ".join(words))
        print(" ".join(words))
        _relation_clean.append(_relation[2])
        relation_clean.append(_relation_clean)

    print(relation_clean)
    write_relations('datasets/relation2id.txt', relation_clean)
