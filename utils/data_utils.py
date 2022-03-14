import codecs
import json
from os.path import join
import pickle
import os
import torch
import numpy as np
import scipy.sparse as sp
from utils import path_file

"""
read raw data
"paper_id": {
    "auhtor_id":{
        "reference_index": 0,
        "title": "",
        "year": 0000,
        "venue": "",
        "affiliation": "",
        "authors": [
            "auhtor_name1", "auhtor_name2", "auhtor_name3", ...
        ],
        "keywords": []
    }
}
["paper_id1", "paper_id2", ...]
"""
def read_raw_data(file_name):
    with open(file_name, 'r', encoding='UTF-8') as json_file:
        raw_data = json.load(json_file)
        return raw_data['pubs'], [raw_data['assignment'][group_id] for group_id in raw_data['assignment']]

# save data
def save(raw_data, file, file_name):
    with open(join(file, file_name + '.json'), 'w') as f:
        json.dump(raw_data, f)
        f.close()

# extract author name, like [name1, name2, ...]
def extract_case_name(file_name):
    case_name = os.path.basename(file_name).split(".json")[0]
    return case_name

# extract publications' id, like [pub_id1, pub_id2, ...]
def extract_pub_id_list(pub_dict):
    pub_id_list = list(pub_dict.keys())
    return pub_id_list

# make list be sparse's coo_matrix
def np_mx_to_sparse_mx(np_mx):
    np_mx = np_mx.reshape((np_mx.shape[0], np_mx.shape[1]))
    return sp.coo_matrix(np_mx,
                         dtype=np.float32)

# make sparse be dense
def sparse_mx_to_torch_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    sparse = torch.sparse.FloatTensor(indices, values, shape)
    dense = sparse.to_dense() # to_dense可以将任何稀疏对象转换回标准密集形式
    # print(dense.size()) # (N, N)
    return dense

# save matrices
def save_gnn_data(
        adj_list,
        edge_label,
        allfeature,
        file_name,
):
    torch.save(
        adj_list,
        '{}/{}'.format(path_file.adj_data_path, file_name)
    )
    torch.save(
        edge_label,
        '{}/{}'.format(path_file.label_data_path, file_name)
    )
    torch.save(
        allfeature,
        '{}/{}'.format(path_file.allfeature_data_path, file_name)
    )

# store graph in the form of triples
def store_graph(file, edge_list):
    with open(file, "w") as f:
        for edge in edge_list:
            f.write(str(edge[0]))
            f.write(' ')
            f.write(str(edge[1]))
            f.write('\n')


# loads make str to dict
def load_json(rfdir, rfname):
    with codecs.open(join(rfdir, rfname), 'r', encoding='utf-8') as rf:
        return json.load(rf)

# dumps makes dict to str
def dump_json(obj, wfpath, wfname, indent=None):
    with codecs.open(join(wfpath, wfname), 'w', encoding='utf-8') as wf:
        json.dump(obj, wf, ensure_ascii=False, indent=indent)

# write binary file
def dump_data(obj, wfpath, wfname):
    with open(os.path.join(wfpath, wfname), 'wb') as wf:
        pickle.dump(obj, wf)

# read binary file
def load_data(rfpath, rfname):
    with open(os.path.join(rfpath, rfname), 'rb') as rf:
        return pickle.load(rf)

# string to json
def serialize_embedding(embedding):
    return pickle.dumps(embedding)

# json to string
def deserialize_embedding(s):
    return pickle.loads(s)