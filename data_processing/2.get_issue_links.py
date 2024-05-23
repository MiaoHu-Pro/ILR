
import os
import random
import json
import time
from utilities import clean
import os
import time
import numpy as np
import pandas as pd
from datetime import datetime

# current : key
# relation [type][inward]
#
# tail : [inwardIssue][key]
#
# {"id": "1957827", "key": "WT-8622", "issuelinks":
#     [{"id": "1165011", "self": "https://jira.mongodb.org/rest/api/2/issueLink/1165011",
#       "type": {"id": "10520", "name": "Problem/Incident",
#                "inward": "is caused by", "outward": "causes",
#                "self": "https://jira.mongodb.org/rest/api/2/issueLinkType/10520"},
#       "inwardIssue": {"id": "1946834", "key": "WT-8543", "self":
#           "https://jira.mongodb.org/rest/api/2/issue/1946834",
#                       "fields": {"summary": "Rollback-to-stable should detect a "
#                                             "prepared transaction as an active "
#                                             "transaction and fail", "status":
#                           {"self": "https://jira.mongodb.org/rest/api/2/status/6",
#                            "description": "The issue is considered finished, the resolution is correct. "
#                                           "Issues which are closed can be reopened.",
#                            "iconUrl": "https://jira.mongodb.org/images/icons/statuses/closed.png",
#                            "name": "Closed", "id": "6", "statusCategory":
#                                {"self": "https://jira.mongodb.org/rest/api/2/statuscategory/3",
#                                 "id": 3, "key": "done", "colorName": "green", "name": "Done"}},
#                                  "priority": {"self": "https://jira.mongodb.org/rest/api/2/priority/3",
#                                               "iconUrl": "https://jira.mongodb.org/images/icons/prioriti"
#                                                          "es/major.svg", "name":
#                                                   "Major - P3", "id": "3"},
#                                  "issuetype": {"self": "https://jira.mongodb.org/rest/api/2/issuetype/1",
#                                                "id": "1", "description": "A problem which impairs"
#                                                                          " or prevents the funct"
#                                                                          "ions of the product.",
#                                                "iconUrl": "https://jira.mongodb.org/secure/viewavatar?size=xsmall&avat"
#                                                           "arId=14703&avatarType=issuetype", "name": "Bug",
#                                                "subtask": false, "avatarId": 14703}}}},
#
#
#      {"id": "1165012", "self": "https://jira.mongodb.org/rest/api/2/issueLink/1165012",
#       "type": {"id": "10012", "name": "Related", "inward": "is related to",
#                "outward": "related to", "self": "https://jira.mongodb.org/rest/api/2/issueLinkType/10012"},
#       "inwardIssue": {"id": "1957826", "key": "WT-8621", "self": "https://jira.mongodb.org/rest/api/2/issue/1957826",
#                       "fields": {"summary": "test_rollback_to_stable30 failed",
#                                  "status": {"self": "https://jira.mongodb.org/rest/api/2/status/3",
#                                             "description": "This issue is being actively worked on at "
#                                                            "the moment by the assignee.", "iconUrl":
#                                                 "https://jira.mongodb.org/images/icons/statuses/inprogress.png",
#                                             "name": "In Progress", "id": "3", "statusCategory": {"self":
#                                                                                                      "https://jira.mongodb.org/rest/api/2/statuscategory/4", "id": 4, "key": "indeterminate", "colorName": "yellow", "name": "In Progress"}}, "priority": {"self": "https://jira.mongodb.org/rest/api/2/priority/3", "iconUrl": "https://jira.mongodb.org/images/icons/priorities/major.svg", "name": "Major - P3", "id": "3"}, "issuetype": {"self": "https://jira.mongodb.org/rest/api/2/issuetype/20", "id": "20", "description": "", "iconUrl": "https://jira.mongodb.org/secure/viewavatar?size=xsmall&avatarId=14703&avatarType=issuetype", "name": "Build Failure", "subtask": false, "avatarId": 14703}}}}], "created": "2022-01-03T22:44:09.000+0000"}
#
#
#
# WT-8622 is caused by WT-8543
# WT-8622 is related to WT-8621


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

def write_issue_issue_links(path,data):
    try:
        fobj = open(path, 'w')

    except IOError as err:
        print('file open error: {0}'.format(err))

    else:
        num = len(data)
        fobj.writelines('%s\n' % num)
        for k in range(len(data)):
            _str = str(data[k][0]) + '\t' + str(data[k][1]) + '\t' + str(data[k][2]) +  '\t' + str(data[k][3]) + '\n'
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

def read_original_issue_links_file(rank_path):

    f = open(rank_path)
    x_obj = []
    key2created = {}
    relation2sub_relation = {}
    i = 0
    for d in f:
        _issue = []
        d = d.strip()
        data = json.loads(d.replace('}{','},{'))
        head = data["key"]
        created_time = data["created"].split("T")


        if created_time[0].split("-")[0] in ["0010","0011","0012","0009","0013"]:
            created_time[0] = '2' + created_time[0][1:]

        created_time = created_time[0] + " " + created_time[1].split(".")[0]


        key2created[head] = created_time
        issue_links = data["issuelinks"]
        if len(issue_links) == 0:
            continue
        for ils in issue_links:
            # print(ils)
            _triple = []
            _relation = ils["type"]["name"]
            if relation2sub_relation.get(_relation) is None:
                relation2sub_relation[_relation] = []
                relation2sub_relation[_relation].append(ils["type"]["inward"])
                relation2sub_relation[_relation].append(ils["type"]["outward"])
            else:
                if ils["type"]["inward"] not in relation2sub_relation[_relation]:
                    relation2sub_relation[_relation].append(ils["type"]["inward"])
                if ils["type"]["outward"] not in relation2sub_relation[_relation]:
                    relation2sub_relation[_relation].append(ils["type"]["outward"])

            outwardIssue = ils.get("outwardIssue")
            if outwardIssue:
                tail = outwardIssue["key"]
            else:
                tail = ils["inwardIssue"]["key"]
            _triple.append(head)
            _triple.append(_relation)
            _triple.append(tail)
            # print(_triple)
            x_obj.append(_triple)
    # print(key2created)


    pure_x_obj = {}
    pure_x_obj_list = []
    for _triple in x_obj:
        _pure_triple = []
        head = _triple[0]
        tail = _triple[2]

        if key2created.get(head) is None or key2created.get(tail) is None:
            continue
        head_with_time = key2created[head]
        tail_with_time = key2created[tail]

        format_pattern = "%Y-%m-%d %H:%M:%S"
        head_time = datetime.strptime(head_with_time , format_pattern)
        tail_time = datetime.strptime(tail_with_time, format_pattern)

        if head_time < tail_time:
            # print(head_time,  tail_time)
            _pure_triple.append(tail)
            _pure_triple.append(_triple[1])
            _pure_triple.append(head)
            _pure_triple.append(tail_with_time)

        else:
            _pure_triple.append(head)
            _pure_triple.append(_triple[1])
            _pure_triple.append(tail)
            _pure_triple.append(head_with_time)
        _str = _pure_triple[0] + '+' + _pure_triple[1] + '+' + _pure_triple[2] + '+' + _pure_triple[3] + '\n'
        if pure_x_obj.get(_str) is None:
            pure_x_obj[_str] = "ok"

            pure_x_obj_list.append(_pure_triple)


    print("pure before", len(x_obj))
    print("after pure",len(pure_x_obj))
    print(len(pure_x_obj_list))
    print(relation2sub_relation)
    f.close()
    return pure_x_obj_list, relation2sub_relation

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

def write_relation_set(path, data):
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
    jira2triples = {}
    file_name = file_name("./jira_issue_links")
    for file in file_name:
        print(file)
        path = "./jira_issue_links/" + file
        x_obj_list, relation2sub_relation = read_original_issue_links_file(path)
        write_issue_issue_links("./issue_links/" + file.split(".")[0].split("_")[0] + "_issue_links.txt", x_obj_list)

        write_relation_set("./relation_set/" + file.split(".")[0] + "_relation_subset.txt", relation2sub_relation)
        jira2triples[file.split("_")[0]] = len(x_obj_list)


        write_entity2id("the_number_of_triples_for_jirs.txt", jira2triples)

