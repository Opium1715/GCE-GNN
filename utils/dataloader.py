from functools import partial
from itertools import chain

import numpy as np
import tensorflow as tf


def generate_data(datas):
    x = datas[0]
    y = datas[1]
    for sample, label in zip(x, y):
        yield sample, label  # 生成几个，流式处理接口就放几个，这里并没有按照 输出定义 输出tuple，而是单个的


def preprocess(raw_data):
    x = raw_data[0]
    label = raw_data[1]
    # 找出最长的序列长度
    len_list = [len(data) for data in x]
    max_len = np.max(len_list)
    # 反转序列 填充0，可以更改为tf2.0写法
    inputs = [list(reversed(upois)) + [0] * (max_len - le) if le < max_len else list(reversed(upois[-max_len:]))
              for upois, le in zip(x, len_list)]
    mask = [[1] * le + [0] * (max_len - le) if le < max_len else [1] * max_len
            for le in len_list]

    inputs = inputs[1222]
    max_n_node = max_len
    node = np.unique(inputs)
    items = node.tolist() + (max_n_node - len(node)) * [0]
    adj = np.zeros((max_n_node, max_n_node))
    for i in np.arange(len(inputs) - 1):
        u = np.where(node == inputs[i])[0][0]
        # 1:自环  2：u->v  3：u<-v  4：u->v， u<-v
        adj[u][u] = 1  # 自环
        if inputs[i + 1] == 0:  # 意味着session就此结束
            break
        v = np.where(node == inputs[i + 1])[0][0]
        if u == v or adj[u][v] == 4:
            continue
        adj[v][v] = 1
        if adj[v][u] == 2:  # 存在v->u，那么u-v都是存在的 置4
            adj[u][v] = 4
            adj[v][u] = 4
        else:
            adj[u][v] = 2
            adj[v][u] = 3
    alias_inputs = [np.where(node == i)[0][0] for i in inputs]
    print(alias_inputs)


def process_data(x, y):  # 这里仅是数据集中的一个元素 (x, y) 流式处理时并不带有batch的维度
    # features = row[0]
    features = x
    labels = y
    items, alias_inputs = tf.unique(features)
    # 注意 alias_inputs 并不一致
    vector_length = tf.shape(features)[0]
    n_nodes = tf.shape(items)[0]
    adj = tf.zeros([n_nodes, n_nodes], dtype=tf.int32)  # TODO: 待会看看需不需要+1 注意shape 留意后续处理
    # A.先算出 value 和 index 然后 创建 稀疏矩阵 转化成 密集矩阵
    # B.如何优化循环，不用python代码
    for i in range(vector_length - 1):
        u = tf.where(condition=items == features[i])[0][0]
        # adj[u][u] = 1
        adj = tf.tensor_scatter_nd_update(tensor=adj, indices=[[u, u]], updates=[1])  # depth = 2
        v = tf.where(condition=items == features[i + 1])[0][0]
        if u == v or adj[u][v] == 4:
            continue
        # adj[v][v] = 1
        adj = tf.tensor_scatter_nd_update(tensor=adj, indices=[[v, v]], updates=[1])
        if adj[v][u] == 2:
            # adj[u][v] = 4
            # adj[v][u] = 4
            adj = tf.tensor_scatter_nd_update(tensor=adj, indices=[[u, v],
                                                                   [v, u]], updates=[4, 4])
        else:
            # adj[u][v] = 2
            # adj[v][u] = 3
            adj = tf.tensor_scatter_nd_update(tensor=adj, indices=[[u, v],
                                                                   [v, u]], updates=[2, 3])
    mask = tf.fill(tf.shape(features), 1.0)
    adj = tf.cast(adj, tf.float32)
    x = (alias_inputs, adj, items, mask, features)
    label = labels - 1
    return x, label


def compute_max_len(raw_data):
    x = raw_data[0]
    # 找出最长的序列长度
    len_list = [len(d) for d in x]
    max_len = np.max(len_list)
    return max_len


def compute_item_num(sequence):
    seq_in_1D = list(chain.from_iterable(sequence))
    items_num = len(np.unique(seq_in_1D))
    return items_num


def process_adj(adj_dict, n_entity, sample_num, num_dict=None):
    # 全局图 随机采样
    adj_entity = np.zeros([n_entity, sample_num], dtype=np.int64)
    num_entity = np.zeros([n_entity, sample_num], dtype=np.int64)
    for entity in range(1, n_entity):
        neighbor = list(adj_dict[entity])
        neighbor_weight = list(num_dict[entity])
        n_neighbor = len(neighbor)
        if n_neighbor == 0:
            continue
        if n_neighbor >= sample_num:
            sampled_indices = np.random.choice(list(range(n_neighbor)), size=sample_num, replace=False)
        else:
            sampled_indices = np.random.choice(list(range(n_neighbor)), size=sample_num, replace=True)
        adj_entity[entity] = np.array([neighbor[i] for i in sampled_indices])
        num_entity[entity] = np.array([neighbor_weight[i] for i in sampled_indices])

    return adj_entity, num_entity


class DataLoader:
    def __init__(self, raw_data, train_mode=True):
        self.max_len = compute_max_len(raw_data)  # 最长序列
        self.data = raw_data
        self.data = self.reverse_data()  # 反转输入序列
        self.train_mode = train_mode
        # self.max_n_node =

    def dataloader(self):
        dataset = tf.data.Dataset.from_generator(generator=partial(generate_data, self.data),
                                                 output_signature=(tf.TensorSpec(shape=None,
                                                                                 dtype=tf.int32),
                                                                   tf.TensorSpec(shape=(),
                                                                                 dtype=tf.int32)))  # (x, label)
        # for data in dataset.batch(1):
        #     print(data)
        #     break
        # dataset = dataset.map(process_data, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(process_data, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)  # 见鬼了，什么奇葩问题
        if self.train_mode:
            pass
            # TODO： 训练时打开shuffle，调试时避免减损性能
            dataset = dataset.shuffle(buffer_size=len(self.data[0]) - (len(self.data[0]) % 100))
        dataset = dataset.padded_batch(batch_size=100,
                                       padded_shapes=(
                                           ([self.max_len],
                                            [self.max_len, self.max_len],
                                            [self.max_len],
                                            [self.max_len],
                                            [self.max_len]
                                            ),
                                           []),
                                       drop_remainder=True
                                       )
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        return dataset

    def reverse_data(self):
        x = self.data[0]
        x = [list(reversed(seq)) for seq in x]
        y = self.data[1]
        new_data = (x, y)
        return new_data


if __name__ == '__main__':
    # path_dataset = '../dataset/tmall'
    # train_data = pickle.load(open(f'{path_dataset}/train.txt', 'rb'))
    a = [903, 907, 906, 905, 904, 903, 902]
    b = 903
    a = tf.constant(a)
    b = tf.constant(b)
    # [903, 907, 906, 905, 904, 903, 902, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    # 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
