import os
import random
import json
import time
from utilities import clean
# from cleantext import clean
import os
import time
import numpy as np
import pandas as pd

from datetime import datetime


def write_issue_obj(path,data):
    try:
        fobj = open(path, 'w')

    except IOError as err:
        print('file open error: {0}'.format(err))

    else:
        num = len(data)

        # fobj.writelines('%s\n' % num)
        for k in range(len(data)):
            _str = str(data[k][0]) + '\t' + str(data[k][1]) + '\t' + str(data[k][2]) +  '\t' + str(data[k][3]) + '\t' \
                   + str(data[k][4]) + '\t' + str(data[k][5]) +  '\t' + str(data[k][6]) + '\t' + str(data[k][7]) + '\n'
            fobj.writelines('%s' % _str)

        fobj.close()
    print('WRITE FILE DONE!')


def read_other_files(rank_path):
    f = open(rank_path)

    x_obj = []
    for d in f:
        d = d.strip()
        if d:
            d = d.split(',')

            elements = []
            for n in d:
                elements.append(n.strip())
            d = elements
            x_obj.append(d)
    f.close()

    return np.array(x_obj)


def read_original_file(rank_path):


    # _each_issue = {'id': id, 'self': self, 'key': key, 'priority': priority, 'issuelinks': issuelinks,
                #                'status': status, 'issuetype': issuetype, 'project': project, 'summary': summary,
                #                'description': description, 'created': created}

    f = open(rank_path)
    x_obj = []
    i = 0
    for d in f:

        _issue = []
        d = d.strip()
        data = json.loads(d.replace('}{','},{'))
        index = data['id']
        print(index)
        syblom = data["key"]

        name = None
        if data.get("summary"):

            name = clean(data["summary"])
        else:
            name = "no name"

        mention = None
        if data.get("description"):

            mention = clean(data["description"])[0:300]
        else:
            mention = name

        project = data["project"]
        issue_type = data["issuetype"]
        status = data["status"]
        _time = data["created"].split("T")

        if _time[0].split("-")[0] in ["0010","0011","0012"]:
            _time[0] = '2' + _time[0][1:]

        _time = _time[0] + " " + _time[1].split(".")[0]

        _issue.append(index)
        _issue.append(syblom)
        _issue.append(" ".join(name))
        _issue.append(project)
        _issue.append(issue_type)
        _issue.append(status)
        _issue.append(" ".join(mention))
        _issue.append(_time)

        x_obj.append(_issue)

        i += 1

    f.close()
    return x_obj


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


def write_issue_2_id(path, data):
    try:
        fobj = open(path, 'w')

    except IOError as err:
        print('file open error: {0}'.format(err))

    else:
        num = len(data)

        fobj.writelines('%s\n' % num)
        for i in range(len(data)):
            _str = str(data[i]) + "\t" + str(i) + '\n'
            fobj.writelines('%s' % _str)

        fobj.close()
    print('WRITE FILE DONE!')


def write_issue(path, data):
    try:
        fobj = open(path, 'w')

    except IOError as err:
        print('file open error: {0}'.format(err))

    else:
        num = len(data)

        fobj.writelines('%s\n' % num)
        for k in range(num):
            _str = str(data[k]) + '\n'
            fobj.writelines('%s' % _str)

        fobj.close()
    print('WRITE FILE DONE!')


def write_pairs(path, data):
    try:
        fobj = open(path, 'w')

    except IOError as err:
        print('file open error: {0}'.format(err))

    else:
        num = len(data)

        fobj.writelines('%s\n' % num)
        for k in range(num):
            _str = str(data[k][0]) + '\t' + str(data[k][1]) + '\t' + str(data[k][2]) + '\n'
            fobj.writelines('%s' % _str)

        fobj.close()
    print('WRITE FILE DONE!')





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


def get_entity_relation_2id(entity_path, relation_path):
    relation = read_files(relation_path)
    entity = read_files(entity_path)

    entity2id_dic = {}
    for i in range(len(entity)):
        entity2id_dic[entity[i][0]] = int(entity[i][1])

    relation2id_dic = {}
    for i in range(len(relation)):
        relation2id_dic[relation[i][0]] = int(relation[i][1])

    # print(entity2id_dic,relation2id_dic)

    return entity2id_dic, relation2id_dic


def write_entity(path, data):
    try:
        fobj = open(path, 'w')

    except IOError as err:
        print('file open error: {0}'.format(err))

    else:
        num = len(data)

        fobj.writelines('%s\n' % num)
        for k in range(num):
            _str = str(data[k]) + '\n'
            fobj.writelines('%s' % _str)

        fobj.close()
    print('WRITE FILE DONE!')


def write_entity2id(path, data):
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

    X = pd.read_csv(entity_obj_path, sep="\t", header=None, dtype=str)

    return np.array(X)


def write_entity_relation_frequency(path, data):
    num = len(data)
    file = open(path, 'w')
    file.writelines('%s\n' % num)
    i = 0
    for e in data:
        file.write(str(i) + '\t' + str(e[0]) + '\t' + str(e[1]) + '\n')
        i += 1

    file.close()


def static_frequency(dic_path, data):
    d = data
    c = dict.fromkeys(d, 0)
    for x in d:
        c[x] += 1
    sorted_x = sorted(c.items(), key=lambda d: d[1], reverse=True)

    num = len(sorted_x)
    file = open(dic_path, 'w')
    file.writelines('%s\n' % num)
    i = 0
    for e in sorted_x:
        file.write(str(i) + '\t' + str(e[0]) + '\t' + str(e[1]) + '\n')
        i += 1

    file.close()


def write_ID_Name_Mention_Time(out_path, data):
    ls = os.linesep
    num = len(data)

    try:
        fobj = open(out_path, 'w')
    except IOError as err:
        print('file open error: {0}'.format(err))

    else:
        for j in range(num):
            #
            _str = str(j) + '\t' + data[j][0] + '\t' + data[j][1] + '\t' + data[j][2] + '\t' + data[j][3] + '\t' + data[j][4] + '\t' + data[j][5] + '\n'
            # _str = str(j) + '\t' + data[j][0] + '\t' + data[j][1] + '\t' + data[j][2] + '\t' + data[j][3] + '\n'
            # _str = str(j) + '\t' + data[j][0] + '\t' + data[j][1] + '\t' + data[j][2] + '\n'

            fobj.writelines('%s' % _str)

        fobj.close()


def cut_mention_fun(mention):
    _s = mention.split(" ")
    if len(_s) <= 100:
        new_s = " ".join(_s)

        return new_s
    else:
        _s = _s[0:100]
        new_s = " ".join(_s)

        return new_s


def file_name(file_dir):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.json':
                L.append(file)
    return L

# file_name_list = file_name(issuesFilePath)

if __name__ == "__main__":

    # id, key, title, project name, issue type, status, description, create time
    # ID_Name_Project_Type_Status_sMention_Time

    file_name = file_name("./jira_issue_information")
    for file in file_name:
        print(file)
        path = "./jira_issue_information/" + file
        new_issue_2_obj = read_original_file(path)
        write_issue_obj("./issue_obj/" + file.split(".")[0].split("_")[0] + "_ID_Name_Project_Type_Status_sMention_Time2.txt", new_issue_2_obj)

    # path = "./jira_issue_information/IntelDAOS_without_issue_link.json"
    # new_issue_2_obj = read_original_file(path)
    # write_issue_obj("./issue_obj/IntelDAOS_ID_Name_Project_Type_Status_sMention_Time.txt", new_issue_2_obj)


    # ID_Name_Mention = read_entity2obj("./data/hadoop_77K/ID_Name_Mention_Project_Type_Time.txt")
    # row, column = ID_Name_Mention.shape
    # print(row, column)

    # id_name_short_mention = []
    # for i in range(row):
    #     _id_name_mention_time = []
    #
    #     Symbol = ID_Name_Mention[i, 1]
    #     Name = ID_Name_Mention[i, 2]
    #     issueProject = ID_Name_Mention[i, 3]
    #     issueType = ID_Name_Mention[i, 4]
    #     Mention = ID_Name_Mention[i, 5]
    #     CreateTime = ID_Name_Mention[i, 6]
    #
    #     if len(Name) == 0 or Name == 'nan':
    #         Name = "None"
    #
    #     if len(Mention) == 0 or Mention == 'nan' or Mention == 'None' or Mention == 'none':
    #         Mention = Name
    #
    #     _id_name_mention_time.append(Symbol)
    #     _id_name_mention_time.append(Name)
    #     _id_name_mention_time.append(issueProject)
    #     _id_name_mention_time.append(issueType)
    #     short_mention = cut_mention_fun(Mention)
    #     _id_name_mention_time.append(short_mention)
    #     _id_name_mention_time.append(CreateTime)
    #
    #     id_name_short_mention.append(_id_name_mention_time)
    #
    # write_ID_Name_Mention_Time("./data/hadoop_77K/ID_Name_sMention_Project_Type_Time.txt", id_name_short_mention)




