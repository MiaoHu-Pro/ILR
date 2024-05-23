# define a class,ERDes,to finish the task of obtaining entity and relation description

# coding:utf-8
import time
import torch
from torch.autograd import Variable
import os
import numpy as np

use_gpu = False


def to_var(x):
    return Variable(torch.from_numpy(x).to(device))


from utilities import read_entity2obj, read_data2id, read_files, construct_entity_des, Enti, read_entity2id, \
    read_ent_rel_2id


def obtain_entity_res(sub_x_obj, _entity_set):

    # 0 id 331989
    # 1 symbol QTWEBSITE-1022
    # 2 name cookie banner
    # 3 projet Qt Project Website
    # 4 type Task
    # 5 status Open
    # 6 des cookie banner pop ups top forms attachment according evermade fix iframe cookie banner form banner iframe control olli apparently manages forward message fine banner id hs eu cookie confirmation style value position change absolute relative
    # 7 create time 2021-12-16 11:55:21

    print("load title and description ...")
    symbol = _entity_set
    all_entity_obj_list = []
    all_entity_description_list = []

    for index in range(len(symbol)):

        entity_id = sub_x_obj[index, 0]
        entity_symbol = sub_x_obj[index, 1]
        entity_name = sub_x_obj[index, 2]

        if entity_name is np.nan:
            entity_name = "no name"

        entity_project = sub_x_obj[index, 3]
        entity_type = sub_x_obj[index, 4]
        entity_status = sub_x_obj[index, 5]

        entity_mention = sub_x_obj[index, 6]
        if entity_mention is np.nan:
            entity_mention = entity_name

        created_time = sub_x_obj[index, 7]

        """"
        to get entity's description
        des = str(self.label/name) + '$' + str(self.description)
        """
        current_entity_des_using_name = []
        entity_des_word_list = construct_entity_des(entity_project, entity_type, entity_name, entity_mention,
                                                    current_entity_des_using_name)

        entity = Enti(_id=entity_id, _symbol=entity_symbol, _project=entity_project, _label=entity_name,
                      _type=entity_type, _status=entity_status, _mention=entity_mention,
                      _neighbours=current_entity_des_using_name,
                      _entity2vec=None,_create_time= created_time,_entity_des_word_list=entity_des_word_list)

        all_entity_obj_list.append(entity)
        all_entity_description_list.append(entity_des_word_list)

    print("load title and description over ...")
    return all_entity_obj_list, all_entity_description_list


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    use_gpu = True


def to_var(x):
    return Variable(torch.from_numpy(x).to(device))


def write_to_file_entity_obj(out_path, all_data):
    ls = os.linesep
    char = " "

    try:
        fobj = open(out_path, 'w')
    except IOError as err:
        print('file open error: {0}'.format(err))

    else:
        for x in all_data:
            # _str = str(x.id) + '\t' + str(x.symbol) + '\t' + str(x.label) + '\t' + str(x.mention) + '\t' + str(
            #     x.neighbours) + '\n'

            _str = str(x.id) + '\t' + str(x.symbol) + '\t' + char.join(x.entity_des) + '\n'
            fobj.writelines('%s' % _str)

        fobj.close()

    print('WRITE FILE DONE!')


def write_triple_descriptions(out_path, head_des, rel_des, tail_des):
    num_triples = len(head_des)
    ls = os.linesep
    head_len = []
    rel_len = []
    tail_len = []
    char = " "

    try:
        fobj = open(out_path, 'w')
    except IOError as err:
        print('file open error: {0}'.format(err))

    else:
        for i in range(num_triples):
            # print(i)
            head = head_des[i]
            rel = rel_des[i]
            tail = tail_des[i]
            head_len.append(len(head))
            rel_len.append(len(rel))
            tail_len.append(len(tail))

            _str = str(i) + '\t' + char.join(head) + '\t' + char.join(rel) + '\t' + char.join(tail) + '\n'
            #
            # _str = str(x.id) + '\t' + str(x.entity_des) + '\n'
            fobj.writelines('%s' % _str)

        fobj.close()

    print('WRITE FILE DONE!')
    print("head len ", np.mean(head_len))
    print("rel len ", np.mean(rel_len))
    print("tail len ", np.mean(tail_len))




class ERDes(object):

    def __init__(self, _Paras):

        self.entity2Obj_path = _Paras['entity2Obj_path']
        self.entity2id_path = _Paras['entity2id_path']
        self.total_triples_with_created_time_path = _Paras['total_triples_with_created_time_path']
        self.training_entity2id_path = _Paras['training_entity2id_path']
        self.relation2id_path = _Paras['relation2id_path']
        self.test_entity2id_path = _Paras['test_entity2id_path']
        self.tokenizer = _Paras['tokenizer']

        self.entity_res = None
        self.entity_symbol_set = []
        self.entity_index_set = []
        self.training_entity_index_set = []
        self.relation2id = []
        self.head_with_relations_and_tail_candidates = None
        self.tail_with_relations_and_head_candidates = None

        # get entity information
        self.issue_obj = read_entity2obj(self.entity2Obj_path)


    def get_entity_des(self):

        print("load entity with textual information ...")
        self.entity_index_set = list(self.issue_obj[:, 0])
        self.entity_symbol_set = list(self.issue_obj[:, 1])
        issue_time = list(self.issue_obj[:, -1])

        symbol2time_dic = {}
        for j in range(len(self.entity_symbol_set)):
            symbol2time_dic[self.entity_symbol_set[j]] = issue_time[j]

        # # obtain_ entity id
        self.entity2id = read_entity2id(self.entity2id_path)

        # self.entity_symbol_set = entity_id_read_file[:, 0].tolist()
        # self.entity_index_set = entity_id_read_file[:, 1].tolist()

        id2symbol_dic = {}
        for i in range(len(self.entity_index_set)):
            id2symbol_dic[self.entity_index_set[i]] = self.entity_symbol_set[i]

        entityid_with_created_time = {}  #
        for i in range(len(self.entity_index_set)):
            entityid_with_created_time[self.entity_index_set[i]] = symbol2time_dic[
                id2symbol_dic[self.entity_index_set[i]]]

        self.entityid_with_created_time = entityid_with_created_time

        # obtain training entity id
        training_id_read_file = read_entity2id(self.training_entity2id_path)
        self.training_entity_index_set = training_id_read_file[:, 1].tolist()

        self.training_entity2id = training_id_read_file

        test_ent_id = read_ent_rel_2id(self.test_entity2id_path)
        self.test_ent_id_index_set = test_ent_id[:, 1].tolist()
        self.test_entity2id = test_ent_id

        # entity with attributes
        all_entity_obj_list, all_entity_description_word_list = obtain_entity_res(self.issue_obj,
                                                                                  self.entity_symbol_set)
        self.entity_with_attributes = all_entity_obj_list
        self.entity_res = {'all_entity_obj_list': all_entity_obj_list,
                           'all_entity_description_word_list': all_entity_description_word_list}

        # relation with attributes
        self.relation2id = read_ent_rel_2id(self.relation2id_path)

        relationid2name = {}
        for i in range(len(self.relation2id)):
            relationid2name[self.relation2id[i][2]] = self.relation2id[i][0]
        self.relationid2name = relationid2name


        relationid2des = {}
        for i in range(len(self.relation2id)):
            relationid2des[self.relation2id[i][2]] = self.relation2id[i][0].split(" ") + self.relation2id[i][1].split(" ")
        self.relationid2des = relationid2des


        print("end entity object with textual information ... \n")

    def get_candidate_entity_relation_pair(self):
        print("begin to get the type of query correspond tail type under a specific  relation")
        total_triples = read_files(self.total_triples_with_created_time_path)
        total_triples_row, column = total_triples.shape

        entity_id = self.issue_obj[:, 0].tolist()
        entity_symbol = self.issue_obj[:, 1].tolist()
        entity_issue_type = self.issue_obj[:, 4].tolist()
        issue_types = list(set(entity_issue_type))

        entity_with_type = {}
        for i in range(len(entity_symbol)):
            entity_with_type[entity_symbol[i]] = entity_issue_type[i]

        head_issue_type_with_relations_and_tail_candidates = {}
        for ist in issue_types:
            head_issue_type_with_relations_and_tail_candidates[ist] = {}

        tail_issue_type_with_relations_and_head_candidates = {}
        for ist in issue_types:
            tail_issue_type_with_relations_and_head_candidates[ist] = {}

        for i in range(total_triples_row):
            head = total_triples[i][0]
            rel = total_triples[i][1]
            tail = total_triples[i][2]

            head_type = entity_with_type[head]
            tail_type = entity_with_type[tail]

            if head_issue_type_with_relations_and_tail_candidates[head_type].get(rel) is None:

                head_issue_type_with_relations_and_tail_candidates[head_type][rel] = set()
                head_issue_type_with_relations_and_tail_candidates[head_type][rel].add(tail_type)
            else:

                head_issue_type_with_relations_and_tail_candidates[head_type][rel].add(tail_type)

            if tail_issue_type_with_relations_and_head_candidates[tail_type].get(rel) is None:

                tail_issue_type_with_relations_and_head_candidates[tail_type][rel] = set()
                tail_issue_type_with_relations_and_head_candidates[tail_type][rel].add(head_type)
            else:
                tail_issue_type_with_relations_and_head_candidates[tail_type][rel].add(head_type)

        self.head_with_relations_and_tail_candidates = head_issue_type_with_relations_and_tail_candidates
        self.tail_with_relations_and_head_candidates = tail_issue_type_with_relations_and_head_candidates
        print("head_issue_type_with_relations_and_tail_candidates", head_issue_type_with_relations_and_tail_candidates)

        training_entity2id = self.training_entity2id.tolist()
        test_entity2id = self.test_entity2id.tolist()


        entityType_with_id = {}
        for ist in list(issue_types):
            entityType_with_id[ist] = set()
        # using training isolated node
        for i in range(len(self.entity2id)):
            entityType_with_id[entity_with_type[self.entity2id[i][0]]].add(self.entity2id[i][1])

        self.entityType_with_id = entityType_with_id



        train_entity_with_type = {}
        for j in range(len(training_entity2id)):
            train_entity_with_type[training_entity2id[j][1]] = entity_with_type[training_entity2id[j][0]]
        self.training_entity_with_type = train_entity_with_type


        test_entity_with_type = {}
        for i in range(len(test_entity2id)):
            test_entity_with_type[test_entity2id[i][1]] = entity_with_type[test_entity2id[i][0]]
        self.test_entity_with_type = test_entity_with_type

        print("end get the type of relation and tail entity ... \n")

    def get_ent_rel_to_tokens(self):

        print("begin get entities and relations' tokens ... ")
        self.ent2tokens = {}
        for i in range(len(self.entity_with_attributes)):
            id = self.entity_with_attributes[i].id
            symbol= self.entity_with_attributes[i].symbol
            created_time = self.entity_with_attributes[i].created_time
            des = self.entity_with_attributes[i].entity_des

            self.ent2tokens[self.entity_with_attributes[i].id] = \
                self.tokenizer.tokenize(" ".join(des))

        self.rel2tokens = {}
        for key, value in  self.relationid2des.items():
            self.rel2tokens[key] = self.tokenizer.tokenize(" ".join(value))
        print("end get entities and relations' tokens ... \n")

    def get_relation_des(self):
        pass

    def get_triple_des(self, _h, _r, _t):
        # print("get triple des begin ... ")

        all_entity_res_obj = self.entity_res['all_entity_obj_list']
        all_entity_des_word = self.entity_res['all_entity_description_word_list']

        head_index = _h
        tail_index = _t
        relation_index = _r

        head_obj = [all_entity_res_obj[i] for i in head_index]
        tail_obj = [all_entity_res_obj[i] for i in tail_index]

        head_description_list = [" ".join(all_entity_des_word[i]) for i in head_index]  # get head entity description

        tail_description_list = [" ".join(all_entity_des_word[i]) for i in tail_index]  # get tail entity

        relation_name = [self.relation2id[i][0] for i in relation_index]

        relation_description_word_list = []
        for i in range(len(relation_name)):
            rel_des = str(relation_name[i]) + ', ' + 'which is between ' + head_obj[i].label + ' and ' + tail_obj[
                i].label + ' .'

            relation_description_word_list.append(rel_des)

        return head_description_list, relation_description_word_list, tail_description_list

    def er_des_print(self):
        print(self.entity2id_path)


def obtain_train_triple_des(file_path, en_rel_des):
    print("obtain_train_triple_des ... \n")
    train_data_set_path = file_path + 'train2id.txt'
    train = read_data2id(train_data_set_path)
    h = train[:, 0].tolist()
    _h = [int(h[i]) for i in range(len(h))]
    t = train[:, 1].tolist()
    _t = [int(t[i]) for i in range(len(t))]
    r = train[:, 2].tolist()
    _r = [int(r[i]) for i in range(len(r))]

    h_des, r_des, t_des = en_rel_des.get_triple_des(_h, _r, _t)

    write_triple_descriptions(file_path + 'train_triple_des_4num_2step.txt', h_des, r_des, t_des)


def obtain_valid_triple_des(file_path, en_rel_des):
    print("obtain_valid_triple_des ... \n")
    valid_data_set_path = file_path + 'valid2id.txt'
    valid = read_data2id(valid_data_set_path)
    h = valid[:, 0].tolist()
    _h = [int(h[i]) for i in range(len(h))]
    t = valid[:, 1].tolist()
    _t = [int(t[i]) for i in range(len(t))]
    r = valid[:, 2].tolist()
    _r = [int(r[i]) for i in range(len(r))]

    h_des, r_des, t_des = en_rel_des.get_triple_des(_h, _r, _t)

    write_triple_descriptions(file_path + 'valid_triple_des_4num_2step.txt', h_des, r_des, t_des)


def obtain_test_triple_des(file_path, en_rel_des):
    print("obtain_test_triple_des ... \n")
    test_data_set_path = file_path + 'test2id.txt'
    test = read_data2id(test_data_set_path)
    h = test[:, 0].tolist()
    _h = [int(h[i]) for i in range(len(h))]
    t = test[:, 1].tolist()
    _t = [int(t[i]) for i in range(len(t))]
    r = test[:, 2].tolist()
    _r = [int(r[i]) for i in range(len(r))]
    h_des, r_des, t_des = en_rel_des.get_triple_des(_h, _r, _t)
    write_triple_descriptions(file_path + 'test_triple_des_4num_2step.txt', h_des, r_des, t_des)


if __name__ == "__main__":
    file_path = '../benchmarks/FB15K237/'
    Paras = {
        'num_neighbours': 4,
        'num_step': 2,
        'word_dim': 100,
        'all_triples_path': file_path + 'train.csv',
        'entity2Obj_path': file_path + 'ID_Name_Mention.txt',
        'entity2id_path': file_path + 'entity2id.txt',
        'relation2id_path': file_path + 'relation2id.txt',
        'entity_des_path': file_path + 'entity2new_des_4nums_2step.txt',
    }
    en_rel_des = ERDes(_Paras=Paras)
    en_rel_des.get_entity_des()

    # train
    obtain_train_triple_des(file_path, en_rel_des)
    # valid
    obtain_valid_triple_des(file_path, en_rel_des)
    # test
    obtain_test_triple_des(file_path, en_rel_des)
