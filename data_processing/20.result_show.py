



import numpy as np
import seaborn as sn
import pandas as pd
import py
import matplotlib.pyplot as plt

def convert2dataframe(recall_rate_results,order):
    data = []

    for method_name in order:
        recall_rate_matrix = recall_rate_results[method_name]
        for run_idx , rr in enumerate(recall_rate_matrix):
            for  k in range(1,21):
                data.append([method_name,run_idx, k, rr[k-1]])

    return pd.DataFrame(data, columns=['method','run', 'k', 'Recall Rate@k'])

def convert2dataframe_diff(recall_rate_results):
    data = []

    for method_name, recall_rate_matrix in recall_rate_results.items():
        recall_rate_matrix = np.round(recall_rate_matrix,2)
        row = [method_name]
        row.extend(recall_rate_matrix)
        data.append(row)

    return pd.DataFrame(data, columns=['Method'] + list(range(1,21)))


# # Jira
# recall_rate_res = {
# 'ILR(BERT)': np.asarray([[0.4043321299638989, 0.4548736462093863, 0.49458483754512633, 0.5258724428399518, 0.5595667870036101, 0.605294825511432, 0.6245487364620939, 0.6510228640192539, 0.6642599277978339, 0.677496991576414, 0.6919374247894103, 0.703971119133574, 0.7160048134777377, 0.7232250300842359, 0.7316486161251504, 0.7460890493381468, 0.7509025270758123, 0.7593261131167268, 0.7653429602888087, 0.7737665463297232, 0.776173285198556]
# ]),
# 'ILR(GPT-2)': np.asarray([[0.5374800637958532, 0.5582137161084529, 0.5885167464114832, 0.6140350877192983, 0.6427432216905901, 0.6889952153110048, 0.7001594896331739, 0.7145135566188198, 0.7240829346092504, 0.7400318979266348, 0.7527910685805422, 0.7607655502392344, 0.7767145135566188, 0.784688995215311, 0.7974481658692185, 0.8022328548644339, 0.8086124401913876, 0.8149920255183413, 0.8245614035087719, 0.8325358851674641, 0.8389154704944178]
# ]),
# }


# Red Hat
# recall_rate_res = {
# 'ILR(BERT)': np.asarray([[0.5582573454913881, 0.5977710233029382, 0.6210739614994935, 0.6453900709219859, 0.6767983789260384, 0.7041540020263425,
#                           0.7183383991894631, 0.723404255319149, 0.7284701114488349, 0.7365754812563323, 0.75177304964539, 0.7578520770010132,
#                           0.7598784194528876, 0.7619047619047619, 0.767983789260385, 0.7740628166160081, 0.7862208713272543, 0.790273556231003,
#                           0.7943262411347518, 0.7963525835866262, 0.7983789260385005]
#
# ]),
# 'ILR(GPT-2)': np.asarray([[0.6089159067882473, 0.6291793313069909, 0.6423505572441742, 0.6605876393110436, 0.6879432624113475, 0.723404255319149,
#                            0.729483282674772, 0.7345491388044579, 0.7527862208713273, 0.7608915906788247, 0.7730496453900709, 0.7811550151975684,
#                            0.7862208713272543, 0.790273556231003, 0.795339412360689, 0.8024316109422492, 0.8085106382978723, 0.8125633232016211,
#                            0.8125633232016211, 0.8156028368794326, 0.8206686930091185]
# ]),
# }

# mojang
recall_rate_res = {
'ILR(BERT)': np.asarray([[0.4052132701421801, 0.5545023696682464, 0.6492890995260664, 0.6943127962085308, 0.7559241706161137,
                          0.7890995260663507, 0.8293838862559242, 0.8436018957345972, 0.8578199052132701, 0.8767772511848341,
                          0.8815165876777251, 0.8933649289099526, 0.9004739336492891, 0.9028436018957346, 0.9052132701421801,
                          0.9075829383886256, 0.9123222748815166, 0.9146919431279621, 0.9170616113744076, 0.9241706161137441,
                          0.9265402843601895]


]),
'ILR(GPT-2)': np.asarray([[0.6079545454545454, 0.7017045454545454, 0.7585227272727273, 0.7926136363636364, 0.8295454545454546, 0.8636363636363636, 0.875, 0.8835227272727273, 0.9005681818181818, 0.9119318181818182, 0.9119318181818182, 0.9232954545454546, 0.9261363636363636, 0.9346590909090909, 0.9403409090909091, 0.9403409090909091, 0.9431818181818182, 0.9431818181818182, 0.9431818181818182, 0.9431818181818182, 0.9460227272727273]
]),
}


# apache
# recall_rate_res = {
# 'ILR(BERT)': np.asarray([[0.32300163132137033, 0.38825448613376834, 0.43230016313213704, 0.46818923327895595, 0.499184339314845, 0.5448613376835236, 0.5709624796084829, 0.5872756933115824, 0.6052202283849919, 0.6182707993474714, 0.6199021207177814, 0.6231647634584013, 0.6378466557911908, 0.6443719412724307, 0.6557911908646004, 0.6639477977161501, 0.6704730831973899, 0.6802610114192496, 0.6818923327895595, 0.6851549755301795, 0.6867862969004894]
# ]),
# 'ILR(GPT-2)': np.asarray([[0.4817073170731707, 0.532520325203252, 0.5447154471544715, 0.5813008130081301, 0.6036585365853658, 0.6341463414634146, 0.6524390243902439, 0.6605691056910569, 0.6686991869918699, 0.676829268292683, 0.6849593495934959, 0.6910569105691057, 0.7012195121951219, 0.7073170731707317, 0.709349593495935, 0.7113821138211383, 0.7195121951219512, 0.7276422764227642, 0.7378048780487805, 0.741869918699187, 0.7459349593495935]
#
#                           ]),
# }

# # # MongoDB
# recall_rate_res = {
# 'ILR(BERT)': np.asarray([[0.46639089968976216, 0.49431230610134436, 0.5108583247156153, 0.5408479834539814,
# 0.5656670113753878, 0.5925542916235781, 0.6070320579110652, 0.6184074457083765, 0.6256463288521199,
# 0.6390899689762151, 0.6514994829369183, 0.6649431230610134, 0.6763185108583247, 0.6794208893485005,
# 0.687728024819028, 0.6949669079627715, 0.7011375387797312, 0.704239917269907, 0.7063081695966908,
# 0.7104788004136504, 0.718717683557394]
# ]),
#
# 'ILR(GPT-2)': np.asarray([[0.4922440537745605, 0.5170630816959669, 0.5305067218200621, 0.5491209927611168, 0.5756670113753878,
# 0.6049638055842813, 0.6194415718717684, 0.6277145811789038, 0.6370217166494312, 0.6473629782833505, 0.6649431230610134,
# 0.6701137538779731, 0.6763185108583247, 0.6814891416752844, 0.688693898655636, 0.6959327817993796, 0.7011375387797312,
# 0.704239917269907, 0.7103764219234746, 0.7195470527404343, 0.7208200620475698]
# ]),
# }
# #



# # #Qt
recall_rate_res = {
'ILR(BERT)': np.asarray([[0.3850806451612903, 0.4102822580645161, 0.4324596774193548, 0.46169354838709675, 0.48286290322580644, 0.5131048387096774, 0.5292338709677419, 0.5413306451612904, 0.5574596774193549, 0.5695564516129032, 0.5796370967741935, 0.5877016129032258, 0.6028225806451613, 0.6159274193548387, 0.6209677419354839, 0.6310483870967742, 0.6391129032258065, 0.6471774193548387, 0.6522177419354839, 0.6592741935483871, 0.6663306451612904]

]),
'ILR(GPT-2)': np.asarray([[0.4084158415841584, 0.4306930693069307, 0.4504950495049505, 0.47277227722772275,
                           0.48886138613861385, 0.525990099009901, 0.5346534653465347, 0.5482673267326733,
                           0.5618811881188119, 0.5742574257425742, 0.5952970297029703, 0.6039603960396039,
                           0.6138613861386139, 0.625, 0.6336633663366337,
                           0.6398514851485149, 0.650990099009901, 0.6584158415841584, 0.6646039603960396,
                           0.6695544554455446, 0.6732673267326733]
])
}
#
sn.set_style("ticks")
# "#00FFFF" "#4C72B0", "#DD8452", , "#937860", "#8C8C8C", "#C44E52", "#FF00FF", "#55A868", "#DD8452" ,
flatui = [ "red", "blue",]
sn.set_palette(sn.color_palette(flatui))

plt.figure(figsize=(9, 6))

order = ["ILR(BERT)", "ILR(GPT-2)"]

order.reverse()

k = list(range(1,21))
ax = sn.lineplot(x="k",y='Recall Rate@k',errorbar='sd',hue='method', style='method', markers=True, dashes=False, data=convert2dataframe(recall_rate_res,order))

import matplotlib.ticker as ticker
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
ax.yaxis.set_major_formatter(ticker.ScalarFormatter())

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles[0:], labels=labels[0:], loc='lower right')

ax.set(ylim=(0.30, 0.75))

fig = ax.get_figure()

fig.savefig("../experimental_result/qt.pdf", format='pdf', bbox_inches='tight', pad_inches=0)

fig.show()





