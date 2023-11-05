import datetime
import pickle
import numpy as np
from itertools import chain


def compute_item_num(sequence):
    seq_in_1D = list(chain.from_iterable(sequence))
    items_num = len(np.unique(seq_in_1D))
    return items_num


dataset = 'tmall'
sample_num = 12
print('数据集：{}'.format(dataset))
print("采样深度：{}".format(sample_num))
print("处理开始于{}".format(datetime.datetime.now()))
f = open(f'dataset/{dataset}/all_train_seq.txt', 'rb')
seq = pickle.load(file=f)
item_num = compute_item_num(seq) + 1  # 40727 + 1 = 40728

relation = []
neighbor = [] * item_num
all_test = set()

adj1 = [dict() for _ in range(item_num)]
adj = [[] for _ in range(item_num)]

for i in range(len(seq)):
    data = seq[i]
    for k in range(1, 4):
        for j in range(len(data) - k):
            relation.append([data[j], data[j + k]])
            relation.append([data[j + k], data[j]])

for tup in relation:
    if tup[1] in adj1[tup[0]].keys():
        adj1[tup[0]][tup[1]] += 1
    else:
        adj1[tup[0]][tup[1]] = 1

weight = [[] for _ in range(item_num)]
for t in range(item_num):
    x = [v for v in sorted(adj1[t].items(), reverse=True, key=lambda x: x[1])]
    adj[t] = [v[0] for v in x]
    weight[t] = [v[1] for v in x]

for i in range(item_num):
    adj[i] = adj[i][:sample_num]
    weight[i] = weight[i][:sample_num]

pickle.dump(adj, open('dataset/' + dataset + '/adj_' + str(sample_num) + '.pkl', 'wb'))
pickle.dump(weight, open('dataset/' + dataset + '/num_' + str(sample_num) + '.pkl', 'wb'))
print('邻接矩阵、边权重、处理完成于{}'.format(datetime.datetime.now()))
