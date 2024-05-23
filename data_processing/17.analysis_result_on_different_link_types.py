



"""
combine these following file:

test.txt
test2id.txt
test_entity2id.txt
test_isolated_node2id.txt
test_relation2id.txt

valid.txt
valid2id.txt
valid_entity2id.txt
valid_isolated_node2id.txt
valid_relation2id.txt

then construct the test-series file:
combine_valid_test_with_latest_head_new_2_new.txt
combine_valid_test_with_latest_head_new_2_old.txt

valid_test_entity2id.txt
valid_test_relation2id.txt
valid_test_isolated2id.txt
"""
import os
import time

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

def write_triples_with_created_date(out_path ,data):

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
            _str = data[j][0] + '\t' + data[j][1] + '\t' + data[j][2 ]+ '\t' + data[j][3] + '\n'

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
        for key ,value in data.items():
            _str = key + "\t" + str(value) + '\n'
            fobj.writelines('%s' % _str)

        fobj.close()
    print('WRITE FILE DONE!')


def read_rank_result(file_path):
    f = open(file_path)
    # f.readline()
    x_obj = []
    for d in f:
        d = d.strip()
        elements = []
        if d:
            d = d.split('-->')

            _test_triple = d[0]
            _test_triple = _test_triple.replace("[", "")
            _test_triple = _test_triple.replace("]", "")
            _test_triple = _test_triple.replace("'", "")


            _test_triple = _test_triple.split(", ")
            for _s in _test_triple:
                elements.append(_s.strip())

            elements.append(int(d[1]))
            elements.append(int(d[3]))
            x_obj.append(elements)
    f.close()

    return x_obj

def read_entity2id(data_id_paht):
    f = open(data_id_paht)
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

if __name__ == "__main__":

    relation_categories = ["general relation", "duplication",
                           "temporal causal", "composition", "workflow"]


    relation2id = read_entity2id("../data/relation2id.txt")

    relation_name_with_id = {}
    for i in range(len(relation2id)):
        relation_name_with_id[relation2id[i][0]] = relation2id[i][2]
    relation_id_with_name = {}
    for i in range(len(relation2id)):
        relation_id_with_name[relation2id[i][2]] = relation2id[i][0]

    file_name = ["redhat_e2_n4_gpt2_relation_tail_prediction_ranks.txt",
                 "Jira_e2_n4_gpt2_relation_tail_prediction_ranks.txt",
                 "mojang_e1_n4_gpt2_relation_tail_prediction_ranks.txt",
                 "apache_e1_n4_gpt2_relation_tail_prediction_ranks.txt",
                 "mongodb_e2_n4_bert_relation_tail_prediction_ranks.txt",
                 "Qt_e2_n4_gpt_relation_tail_prediction_ranks.txt"]

    for _file_name in file_name:

        relation_type_rank = {}
        for relation_type in relation_categories:
            relation_type_rank[relation_type] = []


        print("\n\n\n_file_name: ", _file_name)
        test_triple_res = []

        _test_triple_res = read_rank_result \
        ("../experimental_result/" + _file_name)
        # print("len(test_triple_res):", len(_test_triple_res))
        test_triple_res += _test_triple_res
        # print("len(test_triple_res)", len(test_triple_res))

        # print(relation_type_rank)
        num_of_candidates = []
        for i in range(len(test_triple_res)):

            _rel_id = test_triple_res[i][2]
            _rel_id = _rel_id
            _relation_name = relation_id_with_name[_rel_id]
            relation_type_rank[_relation_name].append(test_triple_res[i][-1])
            num_of_candidates.append(test_triple_res[i][3])

        # print("relation_type_rank",relation_type_rank)

        total_issue_links = []
        relation2_Hits = []
        for relation, rank_list in relation_type_rank.items():
            # print("\n\n relation:", relation)
            time.sleep(1)
            _current_relation2_hits = []
            reak_0 = []
            rank_less_3 = []
            rank_less_5 = []
            rank_less_10 = []
            rank = []
            for i in range(len(rank_list)):
                rank.append(rank_list[i])
                if rank_list[i] == 0:
                    reak_0.append(rank_list[i])
                if rank_list[i] < 3:
                    rank_less_3.append(rank_list[i])
                if rank_list[i] < 5:
                    rank_less_5.append(rank_list[i])
                if rank_list[i] < 10:
                    rank_less_10.append(rank_list[i])


            # print('how many issue links:',len(rank))
            total_issue_links.append(len(rank))
            the_number_of_issue_link = len(rank)

            mr = np.mean(rank)

            new_rank = [ rank[i] + 1 for i in range(len(rank))]
            valid_queries = np.asarray(new_rank)
            MAP_sum = (1 / valid_queries).sum()
            MAP = MAP_sum / valid_queries.shape[0]

            # print("rank: ",rank)
            if len(rank) ==0:
                continue

            hits1 = len(reak_0) / len(rank_list)
            hits3 = len(rank_less_3) / len(rank_list)
            hits5 = len(rank_less_5) / len(rank_list)
            hits10 = len(rank_less_10) / len(rank_list)

            _current_relation2_hits.append(relation)
            _current_relation2_hits.append(the_number_of_issue_link)
            _current_relation2_hits.append(MAP)

            _current_relation2_hits.append(hits1)
            _current_relation2_hits.append(hits3)
            _current_relation2_hits.append(hits5)
            _current_relation2_hits.append(hits10)

            relation2_Hits.append(_current_relation2_hits)

        for i in range(len(relation2_Hits)):

            relation_name = relation2_Hits[i][0]
            relation_mun = relation2_Hits[i][1]
            mr = relation2_Hits[i][2]
            hits1 = relation2_Hits[i][3]
            hits3 = relation2_Hits[i][4]
            hits5 = relation2_Hits[i][5]
            hits10 = relation2_Hits[i][6]

            print("relation_name:{%s}, relation_mun:{%d}, mrr:{%3f}, hits1:{%3f}, hits3:{%3f}, hits5:{%3f}, hits10:{%3f}"%(relation_name,
                                                                                                         relation_mun,mr,hits1,hits3,hits5,hits10))




# redhat_e2_n4_bert_relation_tail_prediction_ranks:
# redhat_e2_n4_gpt2_relation_tail_prediction_ranks.txt


# =======================================
# Jira_e2_n4_bert_relation_tail_prediction_ranks
# Jira_e2_n4_gpt2_relation_tail_prediction_ranks.txt

# =======================================
# Qt_e2_n4_bert_relation_tail_prediction_ranks
# Qt_e2_n4_gpt_relation_tail_prediction_ranks.txt


# =======================================
# Mojang
# mojang_e2_n4_bert_relation_tail_prediction_ranks
# mojang_e1_n4_gpt2_relation_tail_prediction_ranks.txt

# =======================================
# apache_e2_n4_bert_relation_tail_prediction_ranks
# apache_e1_n4_gpt2_relation_tail_prediction_ranks.txt

# =======================================
# mongodb_e2_n4_gpt2_relation_tail_prediction_ranks.txt
# mongodb_e2_n4_bert_relation_tail_prediction_ranks.txt

