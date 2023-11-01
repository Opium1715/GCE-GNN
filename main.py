import pickle
import tensorflow as tf
from utils.dataloader import DataLoader, process_adj

# tf.config.run_functions_eagerly(True)
# tf.data.experimental.enable_debug_mode()
# path_dataset = 'dataset/tmall'
# train_data = pickle.load(open(f'{path_dataset}/train.txt', 'rb'))
# test_data = pickle.load(open(f'{path_dataset}/test.txt', 'rb'))
#
# train_dataloader = DataLoader(train_data, train_mode=True).dataloader()
# test_dataloader = DataLoader(test_data, train_mode=False).dataloader()
adj = pickle.load(open('dataset/tmall/adj_12.pkl', 'rb'))
num = pickle.load(open('dataset/tmall/num_12.pkl', 'rb'))
adj, num = process_adj(adj, n_entity=40728, sample_num=12, num_dict=num)
