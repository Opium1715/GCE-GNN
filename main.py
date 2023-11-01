import pickle
import tensorflow as tf
from utils.dataloader import DataLoader

# tf.config.run_functions_eagerly(True)
# tf.data.experimental.enable_debug_mode()
path_dataset = 'dataset/tmall'
train_data = pickle.load(open(f'{path_dataset}/train.txt', 'rb'))
test_data = pickle.load(open(f'{path_dataset}/test.txt', 'rb'))

train_dataloader = DataLoader(train_data).dataloader()
for data in train_dataloader:
    print(data)
    break
# test_dataloader = DataLoader(test_data)
