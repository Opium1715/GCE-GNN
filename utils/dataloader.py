import pickle
from functools import partial
import pdb
import tensorflow as tf
import numpy as np


def generate_data(datas):
    x = datas[0]
    y = datas[1]
    for sample, label in zip(x, y):
        yield sample, label


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


def process_data(row):  # 这里仅是数据集中的一个元素 (x, y) 流式处理时并不带有batch的维度
    features = row[0]
    labels = row[1]
    items, alias_inputs = tf.unique(features)

    vector_length = tf.shape(features)[0]
    n_nodes = tf.shape(items)[0]
    adj = tf.zeros([n_nodes, n_nodes], dtype=tf.int32)  # 待会看看需不需要+1 注意shape
    adj = tf.Variable(adj)
    for i in range(vector_length - 1):
        u = tf.where(condition=items == features[i])[0][0]
        adj[u][u] = 1
        v = tf.where(condition=items == features[i + 1])[0][0]
        if u == v or adj[u][v] == 4:
            continue
        adj[v][v] = 1
        if adj[v][u] == 2:
            adj[u][v] = 4
            adj[v][u] = 4
        else:
            adj[u][v] = 2
            adj[v][u] = 3
    # indices = tf.gather(alias_inputs, tf.stack([tf.range(vector_length - 1), tf.range(vector_length - 1) + 1],
    #                                            axis=0))
    mask = tf.fill(tf.shape(features), 1)

    x = (alias_inputs, adj, items, mask, features)
    label = labels - 1
    return x, label


def compute_max_len(raw_data):
    x = raw_data[0]
    # 找出最长的序列长度
    len_list = [len(d) for d in x]
    max_len = np.max(len_list)
    return max_len


class DataLoader:
    def __init__(self, data, train_mode=True):
        self.max_len = compute_max_len(data)  # 最长序列
        self.train_mode = train_mode
        # self.max_n_node =

    def dataloader(self, data):
        dataset = tf.data.Dataset.from_generator(generator=partial(generate_data, data),
                                                 output_signature=(tf.TensorSpec(shape=None,
                                                                                 dtype=tf.int32),
                                                                   tf.TensorSpec(shape=(),
                                                                                 dtype=tf.int32)))  # (x, label)
        # for data in dataset.batch(1):
        #     print(data)
        #     break
        # dataset = dataset.map(process_data, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(process_data, num_parallel_calls=tf.data.AUTOTUNE)
        if self.train_mode:
            dataset = dataset.shuffle(buffer_size=int(len(data[0]) / 100))
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


if __name__ == '__main__':
    # path_dataset = '../dataset/tmall'
    # train_data = pickle.load(open(f'{path_dataset}/train.txt', 'rb'))
    a = [903, 907, 906, 905, 904, 903, 902]
    b = 903
    a = tf.constant(a)
    b = tf.constant(b)
    data = (a, b)
    process_data(data)
    # [903, 907, 906, 905, 904, 903, 902, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    # 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
