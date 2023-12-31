import argparse
import datetime
import os
import pickle

from tensorflow import keras

from model.model import GCE_GNN_Model
from utils.dataloader import DataLoader, process_adj, compute_item_num, compute_max_len
from utils.loss import Loss_with_L2
from utils.myCallback import HistoryRecord, P_MRR

# tf.config.run_functions_eagerly(True)
# tf.data.experimental.enable_debug_mode()
parser = argparse.ArgumentParser()
parser.add_argument('--l2', type=float, default=1e-5)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--lr_dc', type=float, default=0.1)
parser.add_argument('--lr_dc_step', type=int, default=3)
parser.add_argument('--n_sample_all', type=int, default=12)
parser.add_argument('--n_sample', type=int, default=12)
parser.add_argument('--n_iter', type=int, default=1)                                    # [1, 2]
parser.add_argument('--dropout_gcn', type=float, default=0, help='Dropout rate.')       # [0, 0.2, 0.4, 0.6, 0.8]
parser.add_argument('--dropout_local', type=float, default=0, help='Dropout rate.')     # [0, 0.5]
parser.add_argument('--dropout_global', type=float, default=0.6, help='Dropout rate.')  # 超参能不能学习呢？
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
opt = parser.parse_args()
print(opt)

path_dataset = 'dataset/tmall'
train_data = pickle.load(open(f'{path_dataset}/train.txt', 'rb'))
test_data = pickle.load(open(f'{path_dataset}/test.txt', 'rb'))
all_data = pickle.load(open(f'{path_dataset}/all_train_seq.txt', 'rb'))
train_data_size = len(train_data[1])
epoch_steps = train_data_size / 100
test_data_size = len(test_data[1])
node_num = compute_item_num(all_data) + 1  # 故意多加一个， 为了和从1开始的编号匹配
seq_len = compute_max_len(train_data)

train_dataloader = DataLoader(train_data, train_mode=True).dataloader()
test_dataloader = DataLoader(test_data, train_mode=False).dataloader()


adj = pickle.load(open('dataset/tmall/adj_12.pkl', 'rb'))
num = pickle.load(open('dataset/tmall/num_12.pkl', 'rb'))
adj, num = process_adj(adj, n_entity=node_num, sample_num=12, num_dict=num)

# model
save_dir = 'logs'
time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
model = GCE_GNN_Model(num_node=node_num, adj_all=adj, num=num, opt=opt, seq_len=seq_len)
lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=opt.lr,
                                                          decay_rate=opt.lr_dc,
                                                          decay_steps=opt.lr_dc_step * epoch_steps,
                                                          staircase=True)
early_stopping = keras.callbacks.EarlyStopping(monitor='MRR@20',
                                               min_delta=0.00005,
                                               patience=3,
                                               verbose=1,
                                               mode='max')
checkpoint_best = keras.callbacks.ModelCheckpoint(filepath=os.path.join(save_dir, 'best_weights.h5'),
                                                  monitor="MRR@20",
                                                  save_weights_only=True,
                                                  save_best_only=True,
                                                  save_freq='epoch')
history_recoder = HistoryRecord(log_dir=os.path.join(save_dir, 'log_' + time_str))
p_mrr = P_MRR(val_data=test_dataloader, performance_mode=2, val_size=int(test_data_size/100))
model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
              loss=Loss_with_L2(model=model, l2=opt.l2, name='scc_loss_with_l2'),
              run_eagerly=False
              )
model.fit(x=train_dataloader,
          epochs=30,
          verbose=1,
          callbacks=[p_mrr, history_recoder, early_stopping],
          validation_data=test_dataloader)  # 注意看！！！
# 后期可以试着改写一下直接评测指标，而不是callback
