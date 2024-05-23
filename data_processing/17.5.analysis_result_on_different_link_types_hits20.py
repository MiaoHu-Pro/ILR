
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

    file_name = [
        # "redhat_e2_n4_gpt2_relation_tail_prediction_ranks.txt",
    #              "Jira_e2_n4_gpt2_relation_tail_prediction_ranks.txt",
    #              "mojang_e1_n4_gpt2_relation_tail_prediction_ranks.txt",
                 "apache_e2_n2_gpt2_relation_tail_prediction_ranks.txt",
    #              "mongodb_e2_n4_gpt2_relation_tail_prediction_ranks.txt",
                 # "Qt_e2_n4_gpt_relation_tail_prediction_ranks.txt"
    ]

    for _file_name in file_name:
        print("\n\n\n_file_name: ", _file_name)
        test_triple_res = []

        relation_type_rank = {}
        for relation_type in relation_categories:
            relation_type_rank[relation_type] = []


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
            rank_down = [[] for i in range(21)]

            # print(len(rank_down), rank_down)

            for i in range(len(rank_list)):
                current = rank_list[i]
                if current <= 20:

                    ll = list([s for s in range(current)])

                    for k in range(len(rank_down)):
                        if k >= current:
                            rank_down[k].append(current)


            # print('how many issue links:',len(rank))
            total_issue_links.append(len(rank_list))
            the_number_of_issue_link = len(rank_list)

            mr = np.mean(rank_list)

            new_rank = [ rank_list[i] + 1 for i in range(len(rank_list))]
            valid_queries = np.asarray(new_rank)
            MAP_sum = (1 / valid_queries).sum()
            MAP = MAP_sum / valid_queries.shape[0]

            # print("rank: ",rank)
            if len(rank_list) ==0:
                continue


            rr = []
            for i in range(len(rank_down)):
                rr.append(len(rank_down[i])/len(rank_list))
                # print(i, len(rank_down[i])/len(rank_list))

            # print(rr)

            _current_relation2_hits.append(relation)
            _current_relation2_hits.append(the_number_of_issue_link)
            _current_relation2_hits.append(MAP)
            for i in range(len(rank_down)):
                _current_relation2_hits.append(float(rr[i]))


            relation2_Hits.append(_current_relation2_hits)

        relation2hists = np.array(relation2_Hits)
        for i in range(len(relation2_Hits)):
            print(relation2hists[i,0],list(map(float, list(relation2hists[i,3:]))))




        # for i in range(len(relation2_Hits)):
        #
        #     relation_name = relation2_Hits[i][0]
        #     relation_mun = relation2_Hits[i][1]
        #     mr = relation2_Hits[i][2]
        #
        #     hits1 = relation2_Hits[i][3]
        #     hits3 = relation2_Hits[i][4]
        #     hits5 = relation2_Hits[i][5]
        #     hits10 = relation2_Hits[i][6]
        #
        #     print("relation_name:{%s}, relation_mun:{%d}, mrr:{%3f}, hits1:{%3f}, hits3:{%3f}, hits5:{%3f}, hits10:{%3f}"%(relation_name,
        #                                                                                                  relation_mun,mr,hits1,hits3,hits5,hits10))




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




"""




_file_name:  redhat_e2_n4_gpt2_relation_tail_prediction_ranks.txt
general relation [0.51953125, 0.53125, 0.5546875, 0.58203125, 0.6015625, 0.6328125, 0.64453125, 0.65234375, 0.6796875, 0.6953125, 0.71875, 0.72265625, 0.73046875, 0.73828125, 0.74609375, 0.7578125, 0.76953125, 0.7734375, 0.7734375, 0.77734375, 0.77734375]
duplication [0.8833333333333333, 0.9066666666666666, 0.9133333333333333, 0.9166666666666666, 0.93, 0.94, 0.9433333333333334, 0.9433333333333334, 0.95, 0.9533333333333334, 0.9566666666666667, 0.9566666666666667, 0.9566666666666667, 0.96, 0.9666666666666667, 0.9666666666666667, 0.9666666666666667, 0.97, 0.97, 0.97, 0.9733333333333334]
temporal causal [0.5017182130584192, 0.5223367697594502, 0.5326460481099656, 0.5498281786941581, 0.5910652920962199, 0.6597938144329897, 0.6632302405498282, 0.6735395189003437, 0.6941580756013745, 0.7010309278350515, 0.7147766323024055, 0.7353951890034365, 0.7353951890034365, 0.7353951890034365, 0.738831615120275, 0.7491408934707904, 0.7491408934707904, 0.7525773195876289, 0.7525773195876289, 0.7560137457044673, 0.7594501718213058]
composition [0.39473684210526316, 0.42105263157894735, 0.43859649122807015, 0.4824561403508772, 0.5263157894736842, 0.5614035087719298, 0.5701754385964912, 0.5701754385964912, 0.5964912280701754, 0.6052631578947368, 0.6140350877192983, 0.6228070175438597, 0.6491228070175439, 0.6491228070175439, 0.6491228070175439, 0.6578947368421053, 0.6842105263157895, 0.6929824561403509, 0.6929824561403509, 0.6929824561403509, 0.7105263157894737]
workflow [0.46153846153846156, 0.5, 0.5, 0.5, 0.5384615384615384, 0.5384615384615384, 0.5384615384615384, 0.5384615384615384, 0.5384615384615384, 0.5384615384615384, 0.5384615384615384, 0.5384615384615384, 0.5384615384615384, 0.5769230769230769, 0.5769230769230769, 0.5769230769230769, 0.5769230769230769, 0.5769230769230769, 0.5769230769230769, 0.6153846153846154, 0.6538461538461539]


_file_name:  Jira_e2_n4_gpt2_relation_tail_prediction_ranks.txt
general relation [0.43853820598006643, 0.4485049833887043, 0.4883720930232558, 0.5249169435215947, 0.5614617940199336, 0.6079734219269103, 0.627906976744186, 0.654485049833887, 0.654485049833887, 0.6710963455149501, 0.6877076411960132, 0.6943521594684385, 0.707641196013289, 0.7209302325581395, 0.7308970099667774, 0.7342192691029901, 0.7408637873754153, 0.7541528239202658, 0.7674418604651163, 0.7774086378737541, 0.7840531561461794]
duplication [0.6735537190082644, 0.7107438016528925, 0.731404958677686, 0.743801652892562, 0.7603305785123967, 0.8057851239669421, 0.8057851239669421, 0.8099173553719008, 0.8264462809917356, 0.8347107438016529, 0.8388429752066116, 0.8512396694214877, 0.8636363636363636, 0.8677685950413223, 0.8842975206611571, 0.8925619834710744, 0.8966942148760331, 0.8966942148760331, 0.9049586776859504, 0.9090909090909091, 0.9132231404958677]
temporal causal [0.5116279069767442, 0.5116279069767442, 0.5348837209302325, 0.5348837209302325, 0.6046511627906976, 0.6976744186046512, 0.6976744186046512, 0.6976744186046512, 0.6976744186046512, 0.7209302325581395, 0.7441860465116279, 0.7441860465116279, 0.7441860465116279, 0.7441860465116279, 0.7674418604651163, 0.7674418604651163, 0.7674418604651163, 0.7674418604651163, 0.7674418604651163, 0.7674418604651163, 0.7906976744186046]
composition [0.47058823529411764, 0.47058823529411764, 0.5294117647058824, 0.5294117647058824, 0.5294117647058824, 0.5294117647058824, 0.5294117647058824, 0.5294117647058824, 0.5882352941176471, 0.6470588235294118, 0.7058823529411765, 0.7058823529411765, 0.8235294117647058, 0.8235294117647058, 0.8235294117647058, 0.8235294117647058, 0.8823529411764706, 0.8823529411764706, 0.8823529411764706, 0.8823529411764706, 0.8823529411764706]
workflow [0.5, 0.5416666666666666, 0.5416666666666666, 0.625, 0.625, 0.625, 0.6666666666666666, 0.6666666666666666, 0.7083333333333334, 0.75, 0.75, 0.75, 0.7916666666666666, 0.7916666666666666, 0.7916666666666666, 0.7916666666666666, 0.7916666666666666, 0.7916666666666666, 0.7916666666666666, 0.8333333333333334, 0.8333333333333334]
0	general relation	5029
1	duplication	3351
2	temporal causal	638
3	workflow	359
4	composition	262


_file_name:  mojang_e1_n4_gpt2_relation_tail_prediction_ranks.txt
/Users/humiao/python_env/language_model_env/lib/python3.7/site-packages/numpy/core/fromnumeric.py:3441: RuntimeWarning: Mean of empty slice.
  out=out, **kwargs)
/Users/humiao/python_env/language_model_env/lib/python3.7/site-packages/numpy/core/_methods.py:189: RuntimeWarning: invalid value encountered in double_scalars
  ret = ret.dtype.type(ret / rcount)
/Users/humiao/python_project/retrieving_relation_tail_for_new_issue_by_plm/data_processing/17.5.analysis_result_on_different_link_types_hits20.py:247: RuntimeWarning: invalid value encountered in double_scalars
  MAP = MAP_sum / valid_queries.shape[0]
general relation [0.42105263157894735, 0.631578947368421, 0.7368421052631579, 0.7368421052631579, 0.7368421052631579, 0.7368421052631579, 0.7368421052631579, 0.7894736842105263, 0.8421052631578947, 0.8421052631578947, 0.8421052631578947, 0.8421052631578947, 0.8421052631578947, 0.8421052631578947, 0.8947368421052632, 0.8947368421052632, 0.9473684210526315, 0.9473684210526315, 0.9473684210526315, 0.9473684210526315, 0.9473684210526315]
duplication [0.6186186186186187, 0.7057057057057057, 0.7597597597597597, 0.7957957957957958, 0.8348348348348348, 0.8708708708708709, 0.8828828828828829, 0.8888888888888888, 0.9039039039039038, 0.9159159159159159, 0.9159159159159159, 0.9279279279279279, 0.9309309309309309, 0.93993993993994, 0.9429429429429429, 0.9429429429429429, 0.9429429429429429, 0.9429429429429429, 0.9429429429429429, 0.9429429429429429, 0.9459459459459459]

0	duplication	32985
1	general relation	1939
2	workflow	36
3	temporal causal	29


g_file_name:  apache_e2_n2_gpt2_relation_tail_prediction_ranks.txt
general relation [0.48148148148148145, 0.5370370370370371, 0.5370370370370371, 0.5787037037037037, 0.5972222222222222, 0.6481481481481481, 0.6712962962962963, 0.6805555555555556, 0.6898148148148148, 0.6944444444444444, 0.7037037037037037, 0.7083333333333334, 0.7222222222222222, 0.7361111111111112, 0.7361111111111112, 0.7407407407407407, 0.75, 0.7546296296296297, 0.7685185185185185, 0.7777777777777778, 0.7824074074074074]
duplication [0.6575342465753424, 0.6712328767123288, 0.684931506849315, 0.684931506849315, 0.6986301369863014, 0.726027397260274, 0.7397260273972602, 0.7397260273972602, 0.7534246575342466, 0.7534246575342466, 0.7534246575342466, 0.7671232876712328, 0.7945205479452054, 0.7945205479452054, 0.7945205479452054, 0.7945205479452054, 0.7945205479452054, 0.7945205479452054, 0.7945205479452054, 0.7945205479452054, 0.7945205479452054]
temporal causal [0.48672566371681414, 0.5398230088495575, 0.5663716814159292, 0.5929203539823009, 0.6283185840707964, 0.6371681415929203, 0.6371681415929203, 0.6460176991150443, 0.6460176991150443, 0.6548672566371682, 0.6548672566371682, 0.6548672566371682, 0.6548672566371682, 0.6548672566371682, 0.6548672566371682, 0.6548672566371682, 0.6637168141592921, 0.6902654867256637, 0.6991150442477876, 0.6991150442477876, 0.6991150442477876]
composition [0.44680851063829785, 0.46808510638297873, 0.46808510638297873, 0.5319148936170213, 0.5319148936170213, 0.5531914893617021, 0.574468085106383, 0.574468085106383, 0.5957446808510638, 0.6170212765957447, 0.6595744680851063, 0.6808510638297872, 0.6808510638297872, 0.6808510638297872, 0.6808510638297872, 0.6808510638297872, 0.7021276595744681, 0.7021276595744681, 0.7021276595744681, 0.7021276595744681, 0.723404255319149]
workflow [0.20930232558139536, 0.32558139534883723, 0.37209302325581395, 0.4418604651162791, 0.4883720930232558, 0.4883720930232558, 0.5348837209302325, 0.5581395348837209, 0.5581395348837209, 0.5813953488372093, 0.5813953488372093, 0.5813953488372093, 0.5813953488372093, 0.5813953488372093, 0.6046511627906976, 0.6046511627906976, 0.6046511627906976, 0.6046511627906976, 0.627906976744186, 0.627906976744186, 0.627906976744186]

5
0	general relation	9955
1	temporal causal	5135
2	duplication	4313
3	composition	2545
4	workflow	1874

_file_name:  mongodb_e2_n4_gpt2_relation_tail_prediction_ranks.txt

general relation [0.44208037825059104, 0.47044917257683216, 0.48936170212765956, 0.5130023640661938, 0.5319148936170213, 0.5791962174940898, 0.6004728132387707, 0.6122931442080378, 0.6217494089834515, 0.6335697399527187, 0.6548463356973995, 0.6619385342789598, 0.6690307328605201, 0.6737588652482269, 0.6855791962174941, 0.6903073286052009, 0.6973995271867612, 0.6997635933806147, 0.7021276595744681, 0.7044917257683215, 0.7115839243498818]
duplication [0.4666666666666667, 0.48333333333333334, 0.49166666666666664, 0.5166666666666667, 0.5416666666666666, 0.5833333333333334, 0.5833333333333334, 0.5833333333333334, 0.5916666666666667, 0.6, 0.6166666666666667, 0.6333333333333333, 0.6333333333333333, 0.6333333333333333, 0.6416666666666667, 0.6583333333333333, 0.6666666666666666, 0.6666666666666666, 0.6833333333333333, 0.7, 0.7166666666666667]
temporal causal [0.47706422018348627, 0.5045871559633027, 0.5107033639143731, 0.5259938837920489, 0.5382262996941896, 0.5779816513761468, 0.5932721712538226, 0.6024464831804281, 0.6116207951070336, 0.6238532110091743, 0.6422018348623854, 0.6422018348623854, 0.6483180428134556, 0.6513761467889908, 0.654434250764526, 0.6636085626911316, 0.6697247706422018, 0.6758409785932722, 0.6788990825688074, 0.6819571865443425, 0.6880733944954128]
composition [0.8115942028985508, 0.8115942028985508, 0.8115942028985508, 0.8115942028985508, 0.8260869565217391, 0.8260869565217391, 0.8260869565217391, 0.8260869565217391, 0.8260869565217391, 0.8260869565217391, 0.8260869565217391, 0.8260869565217391, 0.8405797101449275, 0.8405797101449275, 0.855072463768116, 0.855072463768116, 0.855072463768116, 0.855072463768116, 0.855072463768116, 0.8695652173913043, 0.8695652173913043]
workflow [0.75, 0.7857142857142857, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8928571428571429, 0.8928571428571429, 0.8928571428571429, 0.8928571428571429, 0.8928571428571429, 0.8928571428571429, 0.8928571428571429, 0.8928571428571429, 0.8928571428571429, 0.8928571428571429, 0.8928571428571429, 0.8928571428571429, 0.9285714285714286]

0	general relation	4304
1	temporal causal	3051
2	duplication	1544
3	composition	623
4	workflow	355



_file_name:  Qt_e2_n4_gpt_relation_tail_prediction_ranks.txt
general relation [0.39933993399339934, 0.40594059405940597, 0.41914191419141916, 0.44554455445544555, 0.45874587458745875, 0.49174917491749176, 0.49504950495049505, 0.5016501650165016, 0.5115511551155115, 0.5247524752475248, 0.5412541254125413, 0.5478547854785478, 0.5610561056105611, 0.570957095709571, 0.5775577557755776, 0.5808580858085809, 0.5973597359735974, 0.6105610561056105, 0.6237623762376238, 0.6303630363036303, 0.6303630363036303]
duplication [0.6305732484076433, 0.6496815286624203, 0.6496815286624203, 0.6624203821656051, 0.6751592356687898, 0.732484076433121, 0.7452229299363057, 0.7515923566878981, 0.7643312101910829, 0.7770700636942676, 0.7898089171974523, 0.802547770700637, 0.8089171974522293, 0.8152866242038217, 0.821656050955414, 0.8280254777070064, 0.8343949044585988, 0.8407643312101911, 0.8407643312101911, 0.8407643312101911, 0.8407643312101911]
temporal causal [0.30791788856304986, 0.3460410557184751, 0.3812316715542522, 0.4046920821114369, 0.4252199413489736, 0.4574780058651026, 0.46920821114369504, 0.4897360703812317, 0.5073313782991202, 0.5190615835777126, 0.5483870967741935, 0.5571847507331378, 0.5659824046920822, 0.5777126099706745, 0.5894428152492669, 0.5982404692082112, 0.6070381231671554, 0.6099706744868035, 0.6129032258064516, 0.6187683284457478, 0.6275659824046921]
composition [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
workflow [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

0	temporal causal	2423
1	general relation	2243
2	duplication	1187
3	composition	32
4	workflow	27





"""
