import json
import os
import torch
import random
import numpy as np
import networkx as nx
import scipy.sparse as sp
from utils import path_file
from feature_similarity.Embedding import word_2_vec, load
from feature_similarity.Function import idf
from utils.args import node_feature_size, walk_length
from utils.data_utils import store_graph, read_raw_data, save, load_json

model = load()
IDF = idf()

def read_raw_data(file_name):
    with open(file_name, 'r', encoding='UTF-8') as json_file:
        raw_data = json.load(json_file)
        return raw_data['pubs'], [raw_data['assignment'][group_id] for group_id in raw_data['assignment']]

def extract_case_name(file_name):
    case_name = os.path.basename(file_name).split(".json")[0]
    return case_name

def save_gnn_data(
        node_walk,
        file_name,
        index,
):
    torch.save(
        node_walk,
        '{}/{}'.format(path_file.node_data_path, file_name + '_' + str(index))
    )
    
# class Create_Path(object): 
#     def __init__(self, case_name): 
#         self.G = nx.read_edgelist(os.path.join(path_file.edge_data_path + '/' + case_name + '.txt'), create_using = nx.Graph()) # edges的文件的路径
#         self.walk_len = node_feature_size
#         self.return_prob = 0.4
#         self.multi_info = {}
#         self.generate_multi_info()

#     def generate_multi_info(self):
#         """
#         generate neighbor info 
#         """
#         for node in list(self.G.nodes()):
#             set_2_neighbor = set(self.G.neighbors(node))
#             set_3_neighbor = list()
#             for no in set_2_neighbor:
#                 set_3_neighbor += list(self.G.neighbors(no))
#                 set_3_neighbor = list(set(set_3_neighbor))
#             self.multi_info[node] = set_3_neighbor
        
#     def generate_random_walks(self):
#         """
#         generate random walks
#         """
#         walks = list()
#         for node in list(self.G.nodes()):
#             walk_ = self.random_walk(start=node, node_id=int(node))
#             walk_add_len = self.walk_len - len(walk_)
#             if walk_add_len != 0:
#                 walk_ += [int(node)] * walk_add_len
#             walks.append(walk_)
#         return walks
        
#     def random_walk(self, node_id, rand=random.Random(), start=None):
#         """ 
#             Returns a truncated random walk.
#             alpha: probability of restarts.
#             start: the start node of the random walk.
#         """
#         if start is None:
#             cur = str(node_id)
#         path = []
#         cur = start

#         cur_path_length = self.walk_len
#         neghbor_1 = list(set(self.G.neighbors(cur)))
#         while len(path) < cur_path_length:
#             if len(neghbor_1) > 0:
#                 if rand.random() >= self.return_prob:
#                     path.append(rand.choice(neghbor_1))
#                 else:
#                     path.append(rand.choice(self.multi_info[cur]))
#             else:
#                 break
#         return [[node_id, int(node)] for node in path]

class Create_Path(object): 
    def __init__(self, case_name): 
        self.G = nx.read_edgelist(os.path.join(path_file.edge_data_path + '/' + case_name + '.txt'), create_using = nx.Graph()) # edges的文件的路径
        self.walk_len = node_feature_size
        self.return_prob = 0
        
    def generate_random_walks(self, rand=random.Random(0)):
        """generate random walks
        """
        walks = list()
        node_id = 0
        for node in list(self.G.nodes()):
            walk_ = self.random_walk(rand=rand, start=node, node_id=node_id)
            walk_add_len = self.walk_len - len(walk_)
            if walk_add_len != 0:
                walk_ += [node_id] * walk_add_len
            walks.append(walk_)
            node_id += 1
        return walks
        
    def random_walk(self, node_id, rand=random.Random(), start=None):
        """ Returns a truncated random walk.
            alpha: probability of restarts.
            start: the start node of the random walk.
        """
        if start is None:
            print("random walk need a start node!")
        path = [start]

        cur_path_length = self.walk_len
        while len(path) < cur_path_length:
            cur = path[-1]
            if len(list(self.G.neighbors(cur))) > 0:
                if rand.random() >= self.return_prob:
                    path.append(rand.choice(list(self.G.neighbors(cur))))
                else:
                    path.append(path[0])
            else:
                break
        return [[node_id, int(node)] for node in path]

def node_normalize(matrix):
    """Row-normalize sparse matrix"""
    rowmin = np.min(matrix, 2)[0] # min
    rowmax = np.max(matrix, 2)[0] # max
    rowmin[rowmin == rowmax] = 0
    rowsum = rowmax - rowmin # max - min
    r_inv = (rowsum ** -1).flatten() # 1 / (max - min)
    r_inv[np.isinf(r_inv)] = 0.
    r_inv = r_inv.reshape((-1,1))
    te = matrix.squeeze() - rowmin.reshape((-1,1)) # x - min
    matrix = (r_inv * te) + 0.001 # (x - min) * (max - min)
    return matrix

def context(pub_len, pid_list):
    sentences = load_json(path_file.preprocessed_path, "features.json")

    emb_list = []
    for pid in pid_list:
        sentence = [sentences[pid]]
        emb_list.append(word_2_vec(sentence, model, IDF))

    emb_list = np.array(emb_list).reshape(1, pub_len, -1)
    emb_list = node_normalize(emb_list).reshape(1, pub_len, -1)

    return emb_list

def preprocess(create_Path, allfeature, pub_len, case_name):
    node_feature_list = [[]] * pub_len
    walk = create_Path.generate_random_walks()
    walk = np.array(walk).reshape(-1, 2)
    index = list(zip(*walk))
    node_feature_list = allfeature.squeeze()[tuple(index)]
    node_feature_list = node_feature_list.reshape(1, pub_len, -1)
    node_feature_list = node_normalize(node_feature_list).reshape(1, pub_len, -1)
    # print(node_feature_list)
    # node_feature_list[node_feature_list == 0] = 1
    # for pub_id in range(0, pub_len):
        # node_feature = []
        # for node_id in walk[pub_id]:
            # node_feature.append(allfeature[pub_id][int(node_id)])
        # node_feature_list[pub_id] = np.array(node_feature)
    return node_feature_list

def generate_tripets(case_name):
    feature_matrix = torch.load('{}/{}'.format(path_file.allfeature_data_path, case_name))
    edge_list = np.transpose(np.nonzero(feature_matrix)).tolist()
    store_graph(path_file.edge_data_path + '/' + case_name + '.txt', edge_list)

def preprocess_all():
    for file_index, file_name in enumerate(path_file.train_raw_data_list + path_file.test_raw_data_list):
        case_name = extract_case_name(file_name).lower()
        if case_name in path_file.preprocessed_data_list: # 生成每个author的的处理后的数据
            print('Reading {}... ({}/{})'.format(
                case_name,
                file_index + 1,
                len(path_file.train_raw_data_list + path_file.test_raw_data_list)
            ))
            pub_dict, _ = read_raw_data(file_name)
            pub_len = len(pub_dict)
            create_Path = Create_Path(case_name)
            allfeature = torch.load('{}/{}'.format(path_file.allfeature_data_path, case_name))

            # context_emb = context(pub_len, list(pub_dict.keys()))
            # print("context done")
            
            # save_gnn_data(
            #         context_emb,
            #         case_name,
            #         "context_emb",
            #     )

            for index in range(walk_length):
                processed_data = preprocess(create_Path, allfeature, pub_len, case_name)
                save_gnn_data(
                    *processed_data,
                    case_name,
                    index,
                )
        else:
            print(case_name)

def preprocess_edge():
    for file_index, file_name in enumerate(path_file.train_raw_data_list + path_file.test_raw_data_list):
        case_name = extract_case_name(file_name).lower()
        if case_name in path_file.preprocessed_data_list: # 生成每个author的的处理后的数据
            print('Reading {}... ({}/{})'.format(
                case_name,
                file_index + 1,
                len(path_file.train_raw_data_list + path_file.test_raw_data_list)
            ))
        generate_tripets(case_name)

if __name__ == '__main__':
    preprocess_edge()
    preprocess_all()