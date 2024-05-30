import json
import numpy as np
zinc250_atomic_num_list = [6, 7, 8, 9, 15, 16, 17, 35, 53, 0]
max_atoms = 38
n_bonds = 4


def one_hot_zinc250k(data, out_size=38):
    num_max_id = len(zinc250_atomic_num_list)
    assert tuple(data.shape)[0] == out_size
    b = np.zeros((out_size, num_max_id), dtype=np.float32)
    for i in range(out_size):
        ind = zinc250_atomic_num_list.index(data[i])
        b[i, ind] = 1.0
    return b


def transform_fn_zinc250k(data):
    node, adj, label = data
    node = one_hot_zinc250k(node).astype(np.float32)
    adj = np.concatenate([adj[:3], 1 - np.sum(adj[:3], axis=0, keepdims=
        True)], axis=0).astype(np.float32)
    return node, adj, label


def get_val_ids():
    file_path = '../data/valid_idx_zinc.json'
    print('loading train/valid split information from: {}'.format(file_path))
    with open(file_path) as json_data:
        data = json.load(json_data)
    val_ids = [(idx - 1) for idx in data]
    return val_ids
