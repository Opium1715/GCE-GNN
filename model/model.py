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

        a_input = tf.reshape(tensor=tf.reshape(tensor=tf.tile(h,multiples=tf.constant([1, 1, N])),
                                                shape=(batch_size, N*N, self.dim))
                             * tf.tile(h, multiples=tf.constant([1, N, 1])),
                             shape=(batch_size, N*N, self.dim)
                            )

        e_0 = tf.matmul(a_input, self.a_0)
        e_1 = tf.matmul(a_input, self.a_1)
        e_2 = tf.matmul(a_input, self.a_2)
        e_3 = tf.matmul(a_input, self.a_3)

        e_0 = tf.reshape(tf.squeeze(self.leakyrelu(e_0), axis=-1),(batch_size, N, N))
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
        self.relu = keras.layers.ReLU()
        self.dim = dim
    def build(self, input_shape):
        self.w_1 = self.add_weight(shape=(self.dim+1, self.dim),
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
    def call(self, inputs,  *args, **kwargs):  # inputs = (self_vectors, neighbor_vector, batch_size, masks, neighbor_weight, extra_vector=None)
        self_vectors = inputs[0]
        neighbor_vector = inputs[1]
        batch_size = inputs[2]
        masks = inputs[3]
        neighbor_weight = inputs[4]
        extra_vector = kwargs.get('extra_vector')
        if extra_vector is not None:
            extra_vector = tf.cast(extra_vector,dtype=tf.float32)
            alpha = tf.squeeze(input=tf.matmul(a=tf.concat([tf.tile(input=tf.expand_dims(extra_vector, axis=2),
                                                                    multiples=tf.constant([1, 1, neighbor_vector.shape[2], 1]))
                                                            * neighbor_vector,
                                                            tf.expand_dims(neighbor_weight, axis=-1)],
                                                            axis=-1),
                                                b=self.w_1),
                               axis=-1)
            alpha = tf.nn.leaky_relu(features=alpha, alpha=0.2)
            alpha = tf.squeeze(tf.matmul(alpha, self.w_2), axis=-1)
            alpha = tf.squeeze(tf.nn.softmax(alpha), axis=-1)
            neighbor_vector = tf.reduce_sum(alpha*neighbor_vector, axis=-2)
        else:
            neighbor_vector = tf.reduce_mean(neighbor_vector, axis=2)
        output = tf.concat([self_vectors, neighbor_vector], -1)
        output = self.dropout(output) # 注意training问题
        output = tf.matmul(output, self.w_3)
        output = tf.reshape(batch_size, -1, self.dim)
        output = self.relu(output)

        return output






class GCE_GNN_Model(keras.Model):
    def __init__(self, num_node, adj_all, num):
        super().__init__()
        self.batch_size = 100
        self.num_node = num_node
        self.dim = 100
        self.hop = 1
        self.sample_num = 12
        self.adj_all = tf.constant(adj_all,dtype=tf.int32)
        self.num = tf.constant(num,dtype=tf.float32)

        #Aggregator
        self.local_agg =


    def call(self, inputs, training=None, mask=None):
