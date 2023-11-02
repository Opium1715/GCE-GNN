import pickle
import tensorflow as tf
from tensorflow import keras
from utils.dataloader import DataLoader, process_adj
from model.model import GCE_GNN_Model

# tf.config.run_functions_eagerly(True)
# tf.data.experimental.enable_debug_mode()
opt =
path_dataset = 'dataset/tmall'
train_data = pickle.load(open(f'{path_dataset}/train.txt', 'rb'))
test_data = pickle.load(open(f'{path_dataset}/test.txt', 'rb'))

train_dataloader = DataLoader(train_data, train_mode=True).dataloader()
test_dataloader = DataLoader(test_data, train_mode=False).dataloader()
adj = pickle.load(open('dataset/tmall/adj_12.pkl', 'rb'))
num = pickle.load(open('dataset/tmall/num_12.pkl', 'rb'))
adj, num = process_adj(adj, n_entity=40728, sample_num=12, num_dict=num)

# model
model = GCE_GNN_Model(num_node=40728, adj_all=adj, num=num, opt=opt)
lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=opt.lr,
                                                          decay_rate=opt.lr_dc,
                                                          decay_steps= )
early_stopping = keras.callbacks.EarlyStopping(monitor='MRR@20',
                                               min_delta=0,
                                               patience=4,
                                               verbose=1,
                                               mode='max')
# P_MRR_recoder =
model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
              loss=,
              metrics=[keras.metrics.SparseCategoricalCrossentropy()],
              run_eagerly=False,
              )
model.fit(x = train_dataloader,
          epochs=30,
          verbose=1,
          callbacks=[early_stopping],
          validation_data=test_data)

