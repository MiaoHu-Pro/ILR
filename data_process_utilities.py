import copy
import logging
import os
import random
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from ERDse import ERDes
import networkx as nx
from collections import deque

logger = logging.getLogger(__name__)


class GraphAnalyzer:
    def __init__(self, graph_data):
        """
        Initialize the GraphAnalyzer with the provided graph data.
        """
        self.G = nx.DiGraph()
        self._create_graph(graph_data)
        self.G_undirected = self.G.to_undirected()

    def _create_graph(self, graph_data):
        """
        Create a graph from the provided data.
        """
        for edge in graph_data:
            self.G.add_edge(edge[0], edge[1], label=edge[2])

    def find_neighbors_by_relationship(self, start_node, relationship):
        """
        Find all multi-step neighbors for a given node in the graph (undirected),
        connected through a specific relationship.
        Returns a set of neighbors connected through the specified relationship type.
        """
        visited = set()  # Keep track of visited nodes
        queue = deque([start_node])  # Queue for BFS

        neighbors = set()  # Store neighbors connected through the specified relationship

        while queue:
            current_node = queue.popleft()

            if current_node not in visited:
                visited.add(current_node)

                # Check each neighbor of the current node
                for neighbor in self.G_undirected.neighbors(current_node):
                    if neighbor not in visited:
                        # Check if the edge has the specified relationship
                        if self.G_undirected.get_edge_data(current_node, neighbor)['label'] == relationship:
                            neighbors.add(neighbor)
                            queue.append(neighbor)

        return neighbors


class EntDes(object):
    def __init__(self, _processor, _label_list, _entity_list, _task_name):
        self.processor = _processor
        self.label_list = _label_list
        self.entity_list = _entity_list
        self.task_name = _task_name


class TrainSrc(object):
    def __init__(self, _train_features, _num_train_optimization_steps):
        self.train_features = _train_features
        self.num_train_optimization_steps = _num_train_optimization_steps


class TrainExam(object):
    def __init__(self, _train_example):
        self.train_example = _train_example


class UnseenTestSrc(object):
    def __init__(self, _test_features):
        self.test_features = _test_features


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, text_c=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            text_c: (Optional) string. The untokenized text of the third sequence.
            Only must be specified for sequence triple tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.text_c = text_c
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir, negative):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_train_examples_multi_class(self, data_dir, negative):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_train_examples_two_class(self, data_dir, negative):
        raise NotImplementedError()

    def get_dev_examples(self, data_dir, en_from_set):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self, data_dir):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file.
        reading train.csv will be changed with reading train2id.txt, obtain index
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

    @classmethod
    def _read_isolated_node(cls, input_file, quotechar=None):
        f = open(input_file)
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

        return x_obj


class KGProcessor(DataProcessor):
    """Processor for knowledge graph data set."""

    def __init__(self, args):
        self.labels = set()
        self.en_rel_des = None
        self.args = args
        self.ent_res = None

    def get_entity_res(self, file_path, tokenizer):
        Paras = {
            'tokenizer': tokenizer,
            'total_triples_with_created_time_path': file_path + '/total_triples_with_created_time.txt',
            'entity2Obj_path': file_path + '/ID_Name_Project_Type_Status_sMention_Time.txt',
            'entity2id_path': file_path + '/entity2id.txt',
            'relation2id_path': file_path + '/relation2id.txt',
            'training_entity2id_path': file_path + '/train_entity2id.txt',
            'test_entity2id_path': file_path + '/test_entity2id.txt'
        }


        self.ent_res = ERDes(_Paras=Paras)
        self.ent_res.get_entity_des()
        self.ent_res.get_candidate_entity_relation_pair()
        self.ent_res.get_ent_rel_to_tokens()

    def get_triple_type_info(self):

        return self.ent_res.head_with_relations_and_tail_candidates, self.ent_res.tail_with_relations_and_head_candidates, \
               self.ent_res.entityType_with_id, self.ent_res.relationid2name, self.ent_res.training_entity_with_type, \
               self.ent_res.test_entity_with_type, self.ent_res.entityid_with_created_time

    def get_train_examples(self, data_dir, negative):
        """See base class."""

        return self.to_create_train_examples(
            self._read_tsv(os.path.join(data_dir, "train2id.txt")), "train", data_dir, negative)


    def get_dev_examples(self, data_dir, en_from_set):
        """See base class."""
        # return self._create_examples(
        #     self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev", data_dir)
        return self.to_create_general_examples(
            self._read_tsv(os.path.join(data_dir, "valid2id.txt")), "general_negative_test", data_dir, en_from_set)

    def get_test_general_examples(self, data_dir, test_data, en_from_set, negative):
        print("test2id.txt")
        return self.to_create_general_examples(
            self._read_tsv(os.path.join(data_dir, test_data)), "general_negative_test", data_dir, en_from_set, negative)

    def get_test_isolated_examples(self, data_dir, test_data, en_from_set, negative):
        """See base class."""

        return self.to_create_test_isolated_examples(
            self._read_tsv(os.path.join(data_dir, test_data)), "isolated_triple_test", data_dir, en_from_set,
            negative)

    def get_test_isolated_nodes(self, data_dir, test_data):

        return self._read_isolated_node(os.path.join(data_dir, test_data))

    def get_relation2id(self):
        return self.ent_res.relation2id

    def get_undirecion_graph(self, data_dir):

        graph_data = self._read_tsv(os.path.join(data_dir, "total_triples2id.txt"))

        GAnalyzer = GraphAnalyzer(graph_data)
        return GAnalyzer

    def get_relations(self, data_dir):
        """Gets all labels (relations) in the knowledge graph."""
        # return list(self.labels)
        with open(os.path.join(data_dir, "relations.txt"), 'r') as f:
            lines = f.readlines()
            relations = []
            for line in lines:
                relations.append(line.strip())
        return relations

    def get_labels(self, data_dir):
        """Gets all labels (0, 1) for triples in the knowledge graph."""
        return ["0", "1"]

    def get_entities(self, data_dir):
        """Gets all entities in the knowledge graph."""
        return list(self.ent_res.entity_index_set)

    def get_isolated_and_entities(self):
        """Gets all entities in the knowledge graph."""

        # training_entity_set
        train_entities = self.ent_res.training_entity_index_set
        train_relations = self.ent_res.training_relation_index_set

        # test_entity_set
        test_entities = self.ent_res.test_ent_id_index_set
        test_relations = self.ent_res.test_rel_id_index_set

        # isolated entities
        training_isolated_entities = self.ent_res.training_isolated2id_index_set
        # valid_isolated_entities = self.ent_res.valid_isolated2id_index_set
        test_isolated_entities = self.ent_res.valid_test_isolated2id_index_set

        return train_entities, training_isolated_entities, test_entities, test_isolated_entities

    def get_train_triples(self, data_dir):
        """Gets training triples."""
        return self._read_tsv(os.path.join(data_dir, "train2id.txt"))

    def get_dev_triples(self, data_dir):
        """Gets validation triples."""

        n2n = self._read_tsv(os.path.join(data_dir, "valid_set_new2new2id.txt"))
        n2o = self._read_tsv(os.path.join(data_dir, "valid_set_new2old2id.txt"))
        # return self._read_tsv(os.path.join(data_dir, "valid2id.txt"))
        return n2n + n2o

    def get_test_triples(self, data_dir):
        """Gets test triples."""

        data_test_set = self._read_tsv(os.path.join(data_dir, "test2id.txt"))

        return data_test_set

    def get_test_triples_for_link_prediction(self, data_dir, sub_test_file):
        """Gets test triples."""
        return self._read_tsv(os.path.join(data_dir, sub_test_file))

    def get_time_split_point(self):
        time_split_dict = {"eclipse": "2008-01-01 00:00:00",
                           "mozilla": "2009-12-31 15:38:27",
                           "net_beans": "2008-01-01 00:00:00",
                           "open_office": "2007-12-31 18:40:00",
                           "red_hat": "2019-07-23 00:00:00",
                           "hadoop": "2019-05-3 00:00:00",
                           "mongodb": "2020-02-28 00:00:00"}

        time_split = time_split_dict[self.args.data_dir[7:].strip()]

        return time_split

    def to_create_general_examples(self, lines, set_type, data_dir, en_from_set, negative=1):
        """Creates examples for the training and dev sets."""

        # training_entity_set
        train_entities = list(self.ent_res.training_entity_index_set)
        train_relations = list(self.ent_res.training_relation_index_set)

        # test_entity_set
        test_entities = list(self.ent_res.test_ent_id_index_set)
        test_relations = list(self.ent_res.test_rel_id_index_set)

        if en_from_set == "test_set":
            print("the missing entity come from test set.")
            entities = test_entities
            relations = test_relations

        else:
            print("the missing entity come from train set.")
            entities = train_entities
            relations = train_relations

        lines_str_set = set(['\t'.join(line) for line in lines])
        examples = []
        triples = []
        labels = []

        """
        input sample format: [IDs, head_text, relation_text, tail_text, label]
        """

        for (i, line) in enumerate(lines):

            head_ent_index = line[0]
            tail_ent_index = line[1]
            relation_index = line[2]

            if set_type == "dev" or set_type == "test":

                label = 1
                triples.append([i, int(head_ent_index), int(relation_index), int(tail_ent_index)])
                labels.append(label)

            elif set_type == "unseen_other_graph_rel_test":

                label = 1
                triples.append([i, int(head_ent_index), int(relation_index), int(tail_ent_index)])
                labels.append(label)

            elif set_type == "general_negative_test":

                label = 1
                triples.append([i, int(head_ent_index), int(relation_index), int(tail_ent_index)])
                labels.append(label)

                rnd = random.random()
                guid = "%s-%s" % (set_type + "_corrupt", i)
                if rnd <= 0.5:
                    # corrupting head
                    for j in range(negative):
                        tmp_head = ''
                        while True:
                            tmp_ent_list = set(entities)

                            # tmp_ent_list.remove(line[0])

                            tmp_ent_list = list(tmp_ent_list)
                            tmp_head = random.choice(tmp_ent_list)
                            tmp_triple_str = tmp_head + '\t' + line[1] + '\t' + line[2]
                            if tmp_triple_str not in lines_str_set:
                                break

                        label = 0
                        triples.append([i, int(tmp_head), int(relation_index), int(tail_ent_index)])
                        labels.append(label)
                else:
                    # corrupting tail
                    tmp_tail = ''
                    for j in range(negative):
                        while True:
                            tmp_ent_list = set(entities)

                            # tmp_ent_list.remove(line[1])

                            tmp_ent_list = list(tmp_ent_list)
                            tmp_tail = random.choice(tmp_ent_list)
                            tmp_triple_str = line[0] + '\t' + tmp_tail + '\t' + line[2]
                            # print("tmp_triple_str",tmp_triple_str)
                            if tmp_triple_str not in lines_str_set:
                                break

                        label = 0
                        triples.append([i, int(head_ent_index), int(relation_index), int(tmp_tail)])
                        labels.append(label)

        triple_data = TensorDataset(torch.tensor(triples), torch.tensor(labels, dtype=torch.long))
        triple_dataloader = DataLoader(triple_data, batch_size=2048)
        print("begin to obtain test (general) triple descriptions ... ")
        for step, batch in enumerate(tqdm(triple_dataloader, desc="\n obtain " + set_type + " triple description ")):

            temp_triple, temp_lab = batch
            guid = temp_triple[:, 0]
            head_index = temp_triple[:, 1]

            tail_index = temp_triple[:, 3]

            relation_index = temp_triple[:, 2]

            for i in range(len(temp_lab)):
                examples.append(
                    InputExample(guid=guid[i], text_a=head_index[i], text_b=relation_index[i], text_c=tail_index[i],
                                 label=str(temp_lab[i].cpu().detach().numpy().tolist())))

        print("len(examples)", len(examples))
        return examples

    def to_create_test_triples_for_isolated_nodes(self, lines):
        # 0 : negative
        # 1 : positive
        examples = []
        for (i, line) in enumerate(tqdm(lines)):
            head_ent_index = line[0]
            tail_ent_index = line[1]
            relation_index = line[2]

            examples.append(
                InputExample(guid=i, text_a=head_ent_index, text_b=relation_index, text_c=tail_ent_index,
                             label=str(0)))

        return examples

    def to_create_test_triples_for_positive_nodes(self, lines):
        # 0 : negative
        # 1 : positive
        examples = []
        for (i, line) in enumerate(tqdm(lines)):
            head_ent_index = line[0]
            tail_ent_index = line[1]
            relation_index = line[2]

            examples.append(
                InputExample(guid=i, text_a=head_ent_index, text_b=relation_index, text_c=tail_ent_index,
                             label=str(1)))

        return examples

    def to_create_test_isolated_examples(self, lines, set_type, data_dir, en_from_set, negative=1):
        if set_type == "isolated_triple_test":
            print("constructing test examples including positive and negative.\n")

        # training_entity_set
        train_entities = list(self.ent_res.training_entity_index_set)
        train_relations = list(self.ent_res.training_relation_index_set)

        # test_entity_set
        test_entities = list(self.ent_res.test_ent_id_index_set)
        test_relations = list(self.ent_res.test_rel_id_index_set)

        if en_from_set == "test_set":
            print("the missing entity come from test set.")
            entities = test_entities
            relations = test_relations

        else:
            print("the missing entity come from train set.")
            entities = train_entities
            relations = train_relations

        # isolated entities
        test_isolated_entities = self.ent_res.valid_test_isolated2id_index_set
        train_isolated_entities = self.ent_res.training_isolated2id_index_set
        negative_num = negative * len(lines)

        lines_str_set = set(['\t'.join(line) for line in lines])
        examples = []
        triples = []
        labels = []

        tmp_ent_list = test_isolated_entities + train_isolated_entities + train_entities + test_entities

        """
        input sample:
        a triple [IDs, head_text, relation_text, tail_text, label ]
        """
        for (i, line) in enumerate(lines):
            # if i % 5000 == 0:
            #     print(i)
            head_ent_index = line[0]
            tail_ent_index = line[1]
            relation_index = line[2]

            examples.append(InputExample(guid=i, text_a=head_ent_index, text_b=relation_index, text_c=tail_ent_index,
                                         label=str(0)))

            while True:
                tmp_head = random.choice(tmp_ent_list)
                tmp_triple_str = tmp_head + '' + line[1] + '' + line[2]
                if tmp_triple_str not in lines_str_set:
                    break
            examples.append(InputExample(guid=i, text_a=tmp_head, text_b=relation_index, text_c=tail_ent_index,
                                         label=str(1)))

            while True:
                # tmp_rel = random.choice(relations)
                tmp_tail = random.choice(tmp_ent_list)
                tmp_triple_str = line[0] + '' + tmp_tail + '' + line[2]
                if tmp_triple_str not in lines_str_set:
                    break
            examples.append(InputExample(guid=i, text_a=head_ent_index, text_b=relation_index, text_c=tmp_tail,
                                         label=str(1)))

            while True:
                # tmp_rel = random.choice(relations)
                tmp_head = random.choice(tmp_ent_list)
                tmp_tail = random.choice(tmp_ent_list)
                tmp_triple_str = tmp_head + '' + tmp_tail + '' + line[2]
                if tmp_triple_str not in lines_str_set:
                    break
            examples.append(InputExample(guid=i, text_a=tmp_head, text_b=relation_index, text_c=tmp_tail,
                                         label=str(1)))

            while True:
                # tmp_rel = random.choice(relations)
                tmp_head = random.choice(tmp_ent_list)
                tmp_triple_str = tmp_head + '' + line[1] + '' + line[2]
                if tmp_triple_str not in lines_str_set:
                    break
            examples.append(InputExample(guid=i, text_a=tmp_head, text_b=relation_index, text_c=tail_ent_index,
                                         label=str(1)))

            while True:
                # tmp_rel = random.choice(relations)
                tmp_tail = random.choice(tmp_ent_list)
                tmp_triple_str = line[0] + '' + tmp_tail + '' + line[2]
                if tmp_triple_str not in lines_str_set:
                    break
            examples.append(InputExample(guid=i, text_a=head_ent_index, text_b=relation_index, text_c=tmp_tail,
                                         label=str(1)))

            while True:
                # tmp_rel = random.choice(relations)
                tmp_head = random.choice(tmp_ent_list)
                tmp_tail = random.choice(tmp_ent_list)
                tmp_triple_str = tmp_head + '' + tmp_tail + '' + line[2]
                if tmp_triple_str not in lines_str_set:
                    break
            examples.append(InputExample(guid=i, text_a=tmp_head, text_b=relation_index, text_c=tmp_tail,
                                         label=str(1)))

            # rnd = random.random()
            # guid = "%s-%s" % (set_type + "_corrupt", i)
            # if rnd <= 0.5:
            #     # corrupting head
            #     for j in range(negative):
            #         tmp_head = ''
            #         while True:
            #             tmp_ent_list = set(test_isolated_entities)
            #             # tmp_ent_list.remove(line[0])
            #             tmp_rel = random.choice(relations)
            #             tmp_ent_list = list(tmp_ent_list)
            #             tmp_head = random.choice(tmp_ent_list)
            #             tmp_triple_str = tmp_head + '\t' + line[1] + '\t' + line[2]
            #             # tmp_triple_str = tmp_head + '\t' + line[1] + '\t' + tmp_rel
            #             if tmp_triple_str not in lines_str_set:
            #                 break
            #         examples.append(InputExample(guid=i, text_a=tmp_head, text_b=relation_index, text_c=tail_ent_index,
            #                                      label=str(1)))
            #
            # else:
            #     # corrupting tail
            #     tmp_tail = ''
            #
            #     for j in range(negative):
            #         while True:
            #             tmp_ent_list = set(test_isolated_entities)
            #             # tmp_ent_list.remove(line[1])
            #             tmp_rel = random.choice(relations)
            #             tmp_ent_list = list(tmp_ent_list)
            #             tmp_tail = random.choice(tmp_ent_list)
            #             tmp_triple_str = line[0] + '\t' + tmp_tail + '\t' + line[2]
            #             # print("tmp_triple_str",tmp_triple_str)
            #             # tmp_triple_str = line[0] + '\t' + tmp_tail + '\t' + tmp_rel
            #             if tmp_triple_str not in lines_str_set:
            #                 break
            #         examples.append(InputExample(guid=i, text_a=head_ent_index, text_b=relation_index, text_c=tmp_tail,
            #                                      label=str(1)))

        return examples

    def to_create_test_examples_for_relation_uniform_link_prediction(self, lines, en_from_set):

        examples = []
        # label = 1 positive
        for (i, line) in enumerate(tqdm(lines)):

            if i == 0 or i == 1 or i == 2:
                head_ent_index = line[0]
                tail_ent_index = line[1]
                relation_index = line[2]
                examples.append(
                    InputExample(guid=i, text_a=head_ent_index, text_b=relation_index, text_c=tail_ent_index,
                                 label=str(1)))
            else:

                head_ent_index = line[0]
                tail_ent_index = line[1]
                relation_index = line[2]
                examples.append(
                    InputExample(guid=i, text_a=head_ent_index, text_b=relation_index, text_c=tail_ent_index,
                                 label=str(0)))

        return examples

    def to_create_test_examples_for_link_prediction(self, lines, en_from_set):

        examples = []
        # label = 1 positive
        for (i, line) in enumerate(tqdm(lines)):

            if i == 0:
                head_ent_index = line[0]
                tail_ent_index = line[1]
                relation_index = line[2]
                examples.append(
                    InputExample(guid=i, text_a=head_ent_index, text_b=relation_index, text_c=tail_ent_index,
                                 label=str(1)))
            else:

                head_ent_index = line[0]
                tail_ent_index = line[1]
                relation_index = line[2]
                examples.append(
                    InputExample(guid=i, text_a=head_ent_index, text_b=relation_index, text_c=tail_ent_index,
                                 label=str(0)))

        return examples

    def to_create_test_examples_for_issue_link_detection(self, lines, en_from_set):

        lines_str_set = set(['\t'.join(line) for line in lines])
        examples = []
        triples = []
        labels = []

        for (i, line) in enumerate(tqdm(lines)):
            head_ent_index = line[0]
            tail_ent_index = line[1]
            relation_index = line[2]

            examples.append(
                InputExample(guid=i, text_a=head_ent_index, text_b=relation_index, text_c=tail_ent_index,
                             label=str(1)))

        return examples

    def to_create_train_examples(self, lines, set_type, data_dir, negative=1):
        """Creates examples for the training and dev sets."""
        print("to_create_train_examples")
        # training_entity_set
        tmp_ent_list = list(self.ent_res.training_entity_index_set)

        all_lines_str_set = {}
        for triple in lines:
            triple_str = ''.join(triple)
            all_lines_str_set[triple_str] = "YES"

        examples = []
        """
        1. (s, r, o') tail is replaced by isolated nodes
        2. (s', r, o) head is replaced by isolated nodes
        3. (s', r, o') head and tail are replaced by isolated nodes
        """

        """
        input sample:
        a triple [IDs, head_text, relation_text, tail_text, label]
        """
        # AAAI-2024 method

        # test for relation_uniform

        for (i, line) in enumerate(lines):
            head_ent_index = line[0]
            tail_ent_index = line[1]
            relation_index = line[2]
            # positive
            examples.append(InputExample(guid=i, text_a=head_ent_index, text_b=relation_index, text_c=tail_ent_index,
                                         label=str(1)))
            # AAAI-2024 method
            # # negative
            rnd = random.choice([1, 2, 3])
            if rnd == 1:
                # corrupting head, head come from isolated set
                for j in range(negative):
                    while True:
                        # tmp_rel = random.choice(relations)
                        tmp_head = random.choice(tmp_ent_list)
                        tmp_triple_str = tmp_head + '' + line[1] + '' + line[2]
                        if tmp_triple_str not in all_lines_str_set:
                            break

                    examples.append(InputExample(guid=i, text_a=tmp_head, text_b=relation_index, text_c=tail_ent_index,
                                                 label=str(0)))
            elif rnd == 2:
                # corrupting tail，tail  come from isolated set
                for j in range(negative):
                    while True:

                        # tmp_rel = random.choice(relations)
                        tmp_tail = random.choice(tmp_ent_list)
                        tmp_triple_str = line[0] + '' + tmp_tail + '' + line[2]
                        if tmp_triple_str not in all_lines_str_set:
                            break
                    examples.append(InputExample(guid=i, text_a=head_ent_index, text_b=relation_index, text_c=tmp_tail,
                                                 label=str(0)))
            else:
                # rnd == 3:
                # corrupting tail head, where head and tail come from the isolated set
                for j in range(negative):
                    while True:
                        # tmp_rel = random.choice(relations)
                        tmp_head = random.choice(tmp_ent_list)
                        tmp_tail = random.choice(tmp_ent_list)
                        tmp_triple_str = tmp_head + '' + tmp_tail + '' + line[2]
                        if tmp_triple_str not in all_lines_str_set:
                            break
                    examples.append(InputExample(guid=i, text_a=tmp_head, text_b=relation_index, text_c=tmp_tail,
                                                 label=str(0)))

        return examples

    def to_create_train_examples_multi_class(self, lines, set_type, data_dir, negative=1):

        """Creates examples for the training with multi-classification."""
        # training_entity_set
        training_entities = list(self.ent_res.training_entity_index_set)
        training_relations = list(self.ent_res.training_relation_index_set)

        # isolated entities
        training_isolated_entities = self.ent_res.training_isolated2id_index_set

        all_lines_str_set = {}
        for triple in lines:
            triple_str = ''.join(triple)
            all_lines_str_set[triple_str] = "YES"

        examples = []

        """
        positive :
        (s,r,o)
        negative-1
        (s, r, o') tail is replaced by other linked nodes
        (s', r, o) head is replaced by other linked nodes
        (s', r, o') head and tail are replaced with other linked nodes
        
        negative-2
        (s, r, o') tail is replaced by isolated nodes
        (s', r, o) head is replaced by isolated nodes
        (s', r, o') head and tail are replaced by isolated nodes
        """

        """
        input sample:
        a triple [IDs, head_text, relation_text, tail_text, label ]
        """
        tem_linked_ent_list = list(training_entities)
        tmp_isolated_ent_list = list(training_isolated_entities)

        for (i, line) in enumerate(lines):
            head_ent_index = line[0]
            tail_ent_index = line[1]
            relation_index = line[2]

            examples.append(InputExample(guid=i, text_a=head_ent_index, text_b=relation_index, text_c=tail_ent_index,
                                         label=str(0)))

            """negative-1"""
            # corrupting head, head come from linked set
            for j in range(negative):

                while True:

                    tmp_head = random.choice(tem_linked_ent_list)
                    tmp_triple_str = tmp_head + '' + line[1] + '' + line[2]
                    if tmp_triple_str not in all_lines_str_set:
                        break
                examples.append(InputExample(guid=i, text_a=tmp_head, text_b=relation_index, text_c=tail_ent_index,
                                             label=str(1)))
            # corrupting tail，tail  come from linked set
            for j in range(negative):
                while True:
                    # tmp_rel = random.choice(relations)
                    tmp_tail = random.choice(tem_linked_ent_list)
                    tmp_triple_str = line[0] + '' + tmp_tail + '' + line[2]
                    if tmp_triple_str not in all_lines_str_set:
                        break
                examples.append(InputExample(guid=i, text_a=head_ent_index, text_b=relation_index, text_c=tmp_tail,
                                             label=str(1)))
            # corrupting tail head, where head and tail come from linked set
            for j in range(negative):
                while True:
                    # tmp_rel = random.choice(relations)
                    tmp_head = random.choice(tem_linked_ent_list)
                    tmp_tail = random.choice(tem_linked_ent_list)
                    tmp_triple_str = tmp_head + '' + tmp_tail + '' + line[2]
                    if tmp_triple_str not in all_lines_str_set:
                        break
                examples.append(InputExample(guid=i, text_a=tmp_head, text_b=relation_index, text_c=tmp_tail,
                                             label=str(1)))

            """negative-2"""
            # corrupting head, head come from isolated set
            for j in range(negative):

                while True:
                    # tmp_rel = random.choice(relations)
                    tmp_head = random.choice(tmp_isolated_ent_list)
                    tmp_triple_str = tmp_head + '' + line[1] + '' + line[2]
                    if tmp_triple_str not in all_lines_str_set:
                        break
                examples.append(InputExample(guid=i, text_a=tmp_head, text_b=relation_index, text_c=tail_ent_index,
                                             label=str(2)))
            # corrupting tail，tail  come from isolated set
            for j in range(negative):
                while True:
                    # tmp_rel = random.choice(relations)
                    tmp_tail = random.choice(tmp_isolated_ent_list)
                    tmp_triple_str = line[0] + '' + tmp_tail + '' + line[2]
                    if tmp_triple_str not in all_lines_str_set:
                        break
                examples.append(InputExample(guid=i, text_a=head_ent_index, text_b=relation_index, text_c=tmp_tail,
                                             label=str(2)))
            # corrupting tail head, where head and tail come from isolated set
            for j in range(negative):
                while True:
                    # tmp_rel = random.choice(relations)
                    tmp_head = random.choice(tmp_isolated_ent_list)
                    tmp_tail = random.choice(tmp_isolated_ent_list)
                    tmp_triple_str = tmp_head + '' + tmp_tail + '' + line[2]
                    if tmp_triple_str not in all_lines_str_set:
                        break
                examples.append(InputExample(guid=i, text_a=tmp_head, text_b=relation_index, text_c=tmp_tail,
                                             label=str(2)))

        return examples

    def to_kenize(self, example):

        tokens_a = copy.deepcopy(self.ent_res.ent2tokens[example.text_a])
        tokens_b = copy.deepcopy(self.ent_res.rel2tokens[example.text_b])
        tokens_c = copy.deepcopy(self.ent_res.ent2tokens[example.text_c])

        return (tokens_a, tokens_b, tokens_c)

    def to_convert_examples_to_features(self, examples, label_list, max_seq_length, tokenizer, print_info=True):
        """Loads a data file into a list of `InputBatch`s."""

        label_map = {label: i for i, label in enumerate(label_list)}
        features = []
        sep_token = ""
        first_token = ""
        if self.args.model_type.lower() in ['bert', 'bertone', 'roberta']:
            # print(self.args.model_type.lower())
            sep_token = tokenizer.sep_token
            first_token = tokenizer.cls_token

        elif self.args.model_type.lower() in ['gpt2', 't5', 'mt5', 'llama']:

            sep_token = tokenizer.eos_token
            first_token = ""

        for (ex_index, example) in enumerate(tqdm(examples)):

            if ex_index % 10000 == 0 and print_info:
                logger.info("Writing example %d of %d\n" % (ex_index, len(examples)))

            tokens_a, tokens_b, tokens_c = self.to_kenize(example)

            _truncate_seq_triple(tokens_a, tokens_b, tokens_c, max_seq_length - 4)
            tokens = [first_token] + tokens_a + [sep_token]
            segment_ids = [0] * len(tokens)
            tokens += tokens_b + [sep_token]
            segment_ids += [1] * (len(tokens_b) + 1)
            tokens += tokens_c + [sep_token]
            segment_ids += [0] * (len(tokens_c) + 1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
            input_mask = [1] * len(input_ids)
            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_length - len(input_ids))

            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            label_id = label_map[example.label]

            # if ex_index < 1 and print_info:
            #     logger.info("*** Example ***")
            #     logger.info("guid: %s" % (example.guid))
            #     logger.info("tokens: %s" % " ".join(
            #         [str(x) for x in tokens]))
            #     logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            #     logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            #     logger.info(
            #         "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            #     logger.info("label: %s (id = %d)" % (example.label, label_id))

            features.append(
                InputFeatures(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids, label_id=label_id))

        # print("head plus relation plus tail:  mean(length)", mean(length))
        logger.info("Finish convert_examples_to_features")
        return features


def _truncate_seq_triple(tokens_a, tokens_b, tokens_c, max_length):
    """Truncates a sequence triple in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.

    # total_length = len(tokens_a) + len(tokens_b) + len(tokens_c)
    # if total_length <= max_length:
    #     return True

    while True:
        total_length = len(tokens_a) + len(tokens_b) + len(tokens_c)

        if total_length <= max_length:
            break

        if len(tokens_a) > len(tokens_c):
            tokens_a.pop()
        elif len(tokens_c) > len(tokens_a):
            tokens_c.pop()
        else:
            tokens_c.pop()


# def _truncate_seq_triple(tokens_a, tokens_b, tokens_c, max_length):
#     """Truncates a sequence triple in place to the maximum length."""
#
#     # This is a simple heuristic which will always truncate the longer sequence
#     # one token at a time. This makes more sense than truncating an equal percent
#     # of tokens from each, since if one sequence is very short then each token
#     # that's truncated likely contains more information than a longer sequence.
#
#     # total_length = len(tokens_a) + len(tokens_b) + len(tokens_c)
#     # if total_length <= max_length:
#     #     return True
#
#     while True:
#         total_length = len(tokens_a) + len(tokens_b) + len(tokens_c)
#
#         if total_length <= max_length:
#             break
#         if len(tokens_a) > len(tokens_b) and len(tokens_a) > len(tokens_c):
#             tokens_a.pop()
#         elif len(tokens_b) > len(tokens_a) and len(tokens_b) > len(tokens_c):
#             tokens_b.pop()
#         elif len(tokens_c) > len(tokens_a) and len(tokens_c) > len(tokens_b):
#             tokens_c.pop()
#         else:
#             tokens_c.pop()

def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "kg":
        return {"acc": simple_accuracy(preds, labels)}
    else:
        raise KeyError(task_name)
