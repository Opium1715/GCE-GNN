import tensorflow as tf
from tensorflow import keras
import numpy as np


class LocalAggregator(keras.layers.Layer):
    def __init__(self, dim, alpha, dropout=0.):
        super().__init__()
        self.dim = dim  # shape
        self.dropout = dropout
        self.leakyrelu = keras.layers.LeakyReLU(alpha=alpha)
        self.a_0 = None
        self.a_1 = None
        self.a_2 = None
        self.a_3 = None
        self.bias = None

    def build(self, input_shape):  # TODO: 初始化value？
        self.a_0 = self.add_weight(shape=(self.dim, 1),
                                   dtype=tf.float32,
                                   name='a0')
        self.a_1 = self.add_weight(shape=(self.dim, 1),
                                   dtype=tf.float32,
                                   name='a1')
        self.a_2 = self.add_weight(shape=(self.dim, 1),
                                   dtype=tf.float32,
                                   name='a2')
        self.a_3 = self.add_weight(shape=(self.dim, 1),
                                   dtype=tf.float32,
                                   name='a3')
        self.bias = self.add_weight(shape=(self.dim,),
                                    dtype=tf.float32,
                                    name='bias_localGraph')  # 似乎没用上？

    def call(self, inputs, *args, **kwargs):  # inputs = (hidden, adj)
        h = inputs[0]
        batch_size = tf.shape(h)[0]
        N = tf.shape(h)[1]
        adj = inputs[1]
        mask_item = kwargs.get('mask')
        a_input = tf.reshape(tensor=tf.reshape(tensor=tf.tile(h, multiples=tf.constant([1, 1, N])),
                                               shape=(batch_size, N * N, self.dim))
                                    * tf.tile(h, multiples=tf.constant([1, N, 1])),
                             shape=(batch_size, N * N, self.dim)
                             )

        e_0 = tf.matmul(a_input, self.a_0)
        e_1 = tf.matmul(a_input, self.a_1)
        e_2 = tf.matmul(a_input, self.a_2)
        e_3 = tf.matmul(a_input, self.a_3)

        e_0 = tf.reshape(tf.squeeze(self.leakyrelu(e_0), axis=-1), (batch_size, N, N))
        e_1 = tf.reshape(tf.squeeze(self.leakyrelu(e_1), axis=-1), (batch_size, N, N))
        e_2 = tf.reshape(tf.squeeze(self.leakyrelu(e_2), axis=-1), (batch_size, N, N))
        e_3 = tf.reshape(tf.squeeze(self.leakyrelu(e_3), axis=-1), (batch_size, N, N))

        mask = tf.fill(dims=e_0.get_shapes().dims, value=-9e15)
        alpha = tf.where(condition=tf.equal(adj, 1), x=e_0, y=mask)
        alpha = tf.where(tf.equal(adj, 2), x=e_1, y=alpha)
        alpha = tf.where(tf.equal(adj, 3), x=e_2, y=alpha)
        alpha = tf.where(tf.equal(adj, 4), x=e_3, y=alpha)
        alpha = keras.activations.softmax(alpha)

        return tf.matmul(alpha, h)


class GlobalAggregator(keras.layers.Layer):
    def __init__(self, dim, dropout):
        super().__init__()
        self.bias = None
        self.w_3 = None
        self.w_2 = None
        self.w_1 = None
        self.dropout = keras.layers.Dropout(rate=dropout)
        self.relu = keras.layers.ReLU()  # 还可以切换tanh
        self.dim = dim

    def build(self, input_shape):
        self.w_1 = self.add_weight(shape=(self.dim + 1, self.dim),
                                   dtype=tf.float32,
                                   name='w1')
        self.w_2 = self.add_weight(shape=(self.dim, 1),
                                   dtype=tf.float32,
                                   name='w2')
        self.w_3 = self.add_weight(shape=(2 * self.dim, self.dim),
                                   dtype=tf.float32,
                                   name='w3')
        self.bias = self.add_weight(shape=(self.dim,),
                                    dtype=tf.float32,
                                    name='bias_globalGraph')  # 似乎没用上？

    def call(self, inputs, *args, **kwargs):  # inputs = (self_vectors, neighbor_vector, batch_size, masks,
        # neighbor_weight, extra_vector=None)
        self_vectors = inputs[0]
        neighbor_vector = inputs[1]
        batch_size = inputs[2]
        masks = inputs[3]
        neighbor_weight = inputs[4]
        extra_vector = kwargs.get('extra_vector')
        if extra_vector is not None:
            extra_vector = tf.cast(extra_vector, dtype=tf.float32)
            alpha = tf.squeeze(input=tf.matmul(a=tf.concat([tf.tile(input=tf.expand_dims(extra_vector, axis=2),
                                                                    multiples=tf.constant(
                                                                        [1, 1, neighbor_vector.shape[2], 1]))
                                                            * neighbor_vector,
                                                            tf.expand_dims(neighbor_weight, axis=-1)],
                                                           axis=-1),
                                               b=self.w_1),
                               axis=-1)
            alpha = tf.nn.leaky_relu(features=alpha, alpha=0.2)
            alpha = tf.squeeze(tf.matmul(alpha, self.w_2), axis=-1)
            alpha = tf.squeeze(tf.nn.softmax(alpha), axis=-1)
            neighbor_vector = tf.reduce_sum(alpha * neighbor_vector, axis=-2)
        else:
            neighbor_vector = tf.reduce_mean(neighbor_vector, axis=2)
        output = tf.concat([self_vectors, neighbor_vector], -1)
        output = self.dropout(output)  # 注意training问题
        output = tf.matmul(output, self.w_3)
        output = tf.reshape(output, (batch_size, -1, self.dim))
        output = self.relu(output)

        return output


class GCE_GNN_Model(keras.Model):
    def __init__(self, num_node, adj_all, num, opt):
        super().__init__()
        self.batch_size = 100
        self.num_node = num_node
        self.dim = 100
        self.hop = 1
        self.sample_num = 12
        self.adj_all = tf.constant(adj_all, dtype=tf.int32)
        self.num = tf.constant(num, dtype=tf.float32)

        # Aggregator
        self.local_agg = LocalAggregator(dim=self.dim, alpha=opt.alpha, dropout=0.0)
        self.global_agg = []
        for i in range(self.hop):
            agg = GlobalAggregator(dim=self.dim, dropout=opt.dropout_gcn)
            self.global_agg.append(agg)

        # item $ position encode
        self.embedding = keras.layers.Embedding(self.num_node, self.dim)
        self.pos_embedding = keras.layers.Embedding(200, self.dim)

        # Var
        self.w_1 = self.add_weight(shape=(2 * self.dim, self.dim),
                                   dtype=tf.float32,
                                   name='w_1')
        self.w_2 = self.add_weight(shape=(self.dim, 1),
                                   dtype=tf.float32,
                                   name='w_2')
        self.glu1 = keras.layers.Dense(units=self.dim)  # 注意bias
        self.glu2 = keras.layers.Dense(units=self.dim)
        self.linear_transform = keras.layers.Dense(units=self.dim)

        self.leakyrelu = keras.layers.LeakyReLU(opt.alpha)
        self.dropout_local = keras.layers.Dropout(opt.dropout_local)
        self.dropout_global = keras.layers.Dropout(opt.dropout_global)

    def call(self, inputs, training=None, mask=None):  # inputs = (alias_inputs, items, adj, mask_item, features)
        alias_inputs = inputs[0]
        items = inputs[1]
        batch_size = inputs[1].shape[0]
        seqs_len = inputs[1].shape[1]
        adj = inputs[2]
        mask_item = inputs[3]
        seq_features = inputs[4]

        h = self.embedding(items)

        h_local = self.local_agg(inputs=(h, adj), mask=mask_item)

        item_neighbors = [items]
        weight_neighbors = []
        support_size = seqs_len

        for i in range(1, self.hop + 1):
            item_sample_i = self.adj_all(tf.reshape(item_neighbors[-1], (-1,)))
            weight_sample_i = self.num(tf.reshape(item_neighbors[-1], (-1,)))
            support_size = support_size * self.sample_num
            item_neighbors.append(tf.reshape(item_sample_i, (batch_size, support_size)))
            weight_neighbors.append(tf.reshape(weight_sample_i, (batch_size, support_size)))

        entity_vectors = [self.embedding(i) for i in item_neighbors]
        weight_vectors = weight_neighbors

        session_info = []
        item_emb = self.embedding(seq_features) * tf.expand_dims(mask_item, -1)

        # mean
        sum_item_emb = tf.reduce_sum(item_emb, axis=1) / tf.expand_dims(tf.reduce_sum(mask_item, axis=-1), -1)
        # sum
        sum_item_emb = tf.expand_dims(sum_item_emb, -2)

        for i in range(self.hop):
            session_info.append(tf.tile(sum_item_emb,
                                        multiples=tf.constant(1, entity_vectors[i].shape[1], 1)))

        for n_hop in range(self.hop):
            entity_vectors_next_iter = []
            shape = [batch_size, -1, self.sample_num, self.dim]
            for hop in range(self.hop - n_hop):
                aggregator = self.global_agg[n_hop]
                vector = aggregator(inputs=(entity_vectors[hop],
                                            tf.reshape(entity_vectors[hop + 1], shape=shape),
                                            batch_size,
                                            None,
                                            tf.reshape(weight_vectors[hop], shape=(batch_size, -1, self.sample_num))),
                                    extra_vector=session_info[hop])
                entity_vectors_next_iter.append(vector)
            entity_vectors = entity_vectors_next_iter

        h_global = tf.reshape(entity_vectors[0], (batch_size, seqs_len, self.dim))

        # combine
        h_local = self.dropout_local(h_local)
        h_global = self.dropout_global(h_global)
        h_combine = h_local + h_global

        seq_hidden = tf.stack([h_combine[index][alias_inputs[index]] for index in tf.range(tf.shape(alias_inputs)[0],
                                                                                           dtype=tf.int32)])

        # prediction
        reshape_mask = tf.expand_dims(mask_item, -1)
        batch_size = seq_hidden.shape[0]
        len = seq_hidden.shape[1]
        pos_emb = self.pos_embedding.weights[:len]
        pos_emb = tf.tile(tf.expand_dims(pos_emb, 0), multiples=tf.constant(batch_size, 1, 1))

        hs = tf.reduce_sum(seq_hidden * reshape_mask, -2) / tf.reduce_sum(reshape_mask, 1)
        hs = tf.tile(tf.expand_dims(hs, -2), multiples=tf.constant(1, len, 1))
        nh = tf.matmul(tf.concat([pos_emb, seq_hidden], -1), self.w_1)
        nh = tf.nn.tanh(nh)
        nh = tf.nn.sigmoid(self.glu1(nh) + self.glu2(hs))
        beta = tf.matmul(nh, self.w_2)
        beta = beta * reshape_mask
        select = tf.reduce_sum(beta * seq_hidden, 1)

        b = self.embedding.weights[1:]
        score = tf.matmul(select, b, transpose_b=True)
        output = tf.nn.softmax(score)

        return output


