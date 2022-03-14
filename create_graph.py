import os
import math
import json
import time
import torch
import random
import scipy.io
import numpy as np
import networkx as nx
import multiprocessing
import scipy.sparse as sp
from utils import path_file
from os.path import abspath, dirname, join, exists
from features.Gen_features_similarity import FeatureSimilarityModel
from utils.data_utils import read_raw_data, extract_case_name, np_mx_to_sparse_mx, \
    sparse_mx_to_torch_tensor, save_gnn_data, store_graph, extract_pub_id_list
from utils.args import node_feature_size, time_all

feature_model = FeatureSimilarityModel()

"""extract the positive pairwise publications"""
def extract_positive_pub_pair(assignment):
    positive_pair_list = []
    for group in assignment:
        for index_1, pub_id_1 in enumerate(group):
            for index_2, pub_id_2 in enumerate(group):
                if index_2 < index_1:
                    positive_pair_list.append((pub_id_1, pub_id_2))
    return positive_pair_list

"""extract the negative pairwise publications"""
def extract_negative_pub_pair(annotation_result):
    negative_pair_list = []
    for index_1, group_1 in enumerate(annotation_result): # [[], [], [], [], []]
        for index_2, group_2 in enumerate(annotation_result):
            if index_2 < index_1:
                for pub_id_1 in group_1:
                    for pub_id_2 in group_2:
                        negative_pair_list.append((pub_id_1, pub_id_2))
    return negative_pair_list

def preprocess(pub_dict, assignment, case_name):  
    pub_id_list = extract_pub_id_list(pub_dict) # achieve paper_id
    raw_pub_list = [pub_dict[pub_id] for pub_id in pub_id_list] # achieve the content of papers
    N = len(pub_id_list)
    print('Document num: {}'.format(N))
    edge_feature_list = np.array(feature_model.cal_pairwise_sim(raw_pub_list)) # the feature of edge
    n_edge_features = edge_feature_list.shape[-1] # the number of different features : 6
    edge_feature_list = edge_feature_list.reshape([N, N, n_edge_features]) # (N, N, 6)
    edge_feature_list = np.split(edge_feature_list, n_edge_features, axis=2) # [(N, N) * 6]
    edge_feature_list = list(map(lambda x: x.squeeze(), edge_feature_list)) # [(N, N) * 6]

    edge_feature_list_ = []
    all_feature = []
    print_ = ["coauthor_sim", "affiliation_sim", "venue_sim", "title_sim", "keywords_sim", "all_feature_sim"]
    for edge_feature_index, feature_matrix in enumerate(edge_feature_list):
        # print(print_[edge_feature_index])

        # feature_matrix = np.tril(feature_matrix, k=-1) + np.triu(feature_matrix, k=1) # 去除对角线值
        # feature_max = np.nanmax(feature_matrix, axis=1) + 0.1 # 求一行最大值

        # ones_nan = np.ones_like(feature_max) + np.nan
        # ones_nan = np.diag(ones_nan)
        # feature_mean = np.nanmean(feature_matrix + ones_nan, axis=1) # 求一行的均值
        # if edge_feature_index > 4:
        #     feature_mean = (feature_mean * time_all + feature_max) / (time_all + 1)

        # feature_max[feature_max < 1] = 1
        # feature_matrix += np.diag(feature_max) # 加上对角值
        # feature_matrix[np.isnan(feature_matrix)] = 0 # 将nan 赋值为0

        # feature_matrix[feature_matrix < feature_mean + 1e-14] = 0 # 去除小于均值的数值
        # diag = np.diag(feature_matrix)
        # feature_matrix = np.tril(feature_matrix, k=-1) # 提取下三角矩阵
        # feature_matrix += np.transpose(feature_matrix) # 矩阵进行转置
        # feature_matrix += np.diag(diag)
        # feature_matrix[np.isnan(feature_matrix)] = 0
        # feature_matrix = np.tril(feature_matrix, k=-1) + np.triu(feature_matrix, k=1) # 去除对角线值
        # feature_max = np.max(feature_matrix, axis=1) + 0.2
        # diag_ = np.copy(feature_max)
        # diag_[feature_max < 1] = 1
        # feature_matrix += np.diag(diag_)
        # feature_matrix += np.transpose(feature_matrix)
        # feature_matrix /= 2
        # print(feature_matrix)
        feature_mean = np.nanmean(feature_matrix, axis=0) # 忽略这里面的nan，进行求均值1
        feature_matrix[np.isnan(feature_matrix)] = 0 # 将nan 赋值为0
        thread = 10
        if np.where(feature_matrix > thread)[0].shape[0] != 0:
            feature_matrix = np.log(feature_matrix + 1)
            feature_mean = np.log(feature_mean + 1)
        if edge_feature_index > 4:
            feature_mean = (time_all * feature_mean + np.max(feature_matrix, axis=0)) / (time_all + 1)
        feature_matrix[feature_matrix < feature_mean + 1e-4] = 0 # 进行剪枝操作
        diag = np.copy(np.diag(feature_matrix)) # 形成对角矩阵
        diag_ = np.copy(diag)
        diag_[diag == 0] = 1
        feature_matrix -= np.diag(diag) # 去掉对角元素
        feature_matrix += np.diag(diag_) # 用1 进行填充
        feature_matrix += np.transpose(feature_matrix) # 矩阵进行转置
        feature_matrix /= 2 # 对原矩阵进行归一话操作

        if edge_feature_index <= 4:
            edge_feature_list_.append(feature_matrix)
        else:
            all_feature = feature_matrix

    edge_feature_list = edge_feature_list_

    adj_list = list(map(np_mx_to_sparse_mx, edge_feature_list))
    adj_list = list(map(sparse_mx_to_torch_tensor, adj_list))

    positive_pub_id_pair_list = extract_positive_pub_pair(assignment)
    negative_pub_id_pair_list = extract_negative_pub_pair(assignment)

    print('Generating edge labels...')
    label_dict = {"{}_{}".format(pub_id_1, pub_id_2): 1 for pub_id_1, pub_id_2 in positive_pub_id_pair_list}
    label_dict.update(
        {"{}_{}".format(pub_id_2, pub_id_1): 1 for pub_id_1, pub_id_2 in positive_pub_id_pair_list})
    label_dict.update(
        {"{}_{}".format(pub_id_1, pub_id_2): 0 for pub_id_1, pub_id_2 in negative_pub_id_pair_list})
    label_dict.update(
        {"{}_{}".format(pub_id_2, pub_id_1): 0 for pub_id_1, pub_id_2 in negative_pub_id_pair_list})
    edge_label = torch.zeros_like(adj_list[0]).long()

    for col_index in range(0, N):
        for row_index in range(0, N):
            if col_index != row_index:
                pub_id_1 = pub_id_list[col_index]
                pub_id_2 = pub_id_list[row_index]
                edge_label[col_index][row_index] = label_dict["{}_{}".format(pub_id_1, pub_id_2)]
            else:
                edge_label[col_index][row_index] = 1

    return adj_list, edge_label, all_feature

class DataDealProcess(multiprocessing.Process):
    def __init__(self, queue=None):
        super().__init__()
        self.queue = queue

    def run(self):
        while True:
            batch = self.queue.get()
            try:
                self.process_data(batch)
            except Exception as ex:
                print(ex)
                time.sleep(5)
                self.process_data(batch)
            self.queue.task_done()

    def process_data(self, batch):
        for file_name in batch:
            preprocess_one_paper(file_name)
        print("OK")

def DealWithProcess(author_list, num_processes=multiprocessing.cpu_count(), queue_limit=10):
    batch_size = math.ceil(float(len(author_list)) / queue_limit)

    with multiprocessing.Manager() as manager:  
        queue = multiprocessing.JoinableQueue() 
        workerList = []

        for i in range(num_processes):
            worker = DataDealProcess(queue=queue)
            workerList.append(worker)
            worker.daemon = True 
            worker.start() 

        for i in range(0, batch_size - 1):
            sub_list = author_list[i * queue_limit: (i + 1) * queue_limit]
            queue.put(sub_list)
        queue.put(author_list[(batch_size - 1) * queue_limit:])
        queue.join()

        for worker in workerList:
            worker.terminate()

"""generate the data with single process"""
def preprocess_all():
    for file_index, file_name in enumerate(path_file.train_raw_data_list + path_file.test_raw_data_list):
        case_name = extract_case_name(file_name).lower() 
        if case_name not in path_file.preprocessed_data_list: # 生成每个author的的处理后的数据
            print('Reading {}... ({}/{})'.format(
                case_name,
                file_index + 1,
                len(path_file.train_raw_data_list + path_file.test_raw_data_list)
            ))
            start = time.time()
            processed_data = preprocess(*read_raw_data(file_name), case_name)
            save_gnn_data(
                *processed_data,
                case_name,
            )
            print(time.time() - start)

def preprocess_one_paper(file_name):
    case_name = extract_case_name(file_name).lower()
    if case_name not in path_file.preprocessed_data_list: # 生成每个author的的处理后的数据
        print('Reading {}...'.format(case_name))
        processed_data = preprocess(*read_raw_data(file_name), case_name)
        save_gnn_data(*processed_data, case_name,)

"""generate the data with multi-processing"""
def deal_all_with_mul():
    root_file_name_list = []

    for file_index, file_name in enumerate(path_file.train_raw_data_list + path_file.test_raw_data_list):
        case_name = extract_case_name(file_name).lower()  # extract author name

        if os.path.exists(join(path_file.adj_data_path, case_name)) == True:
            print("{0} has done.".format(case_name))
            continue
        else:
            root_file_name_list.append(file_name)

    print(len(root_file_name_list))

    DealWithProcess(root_file_name_list, num_processes=20, queue_limit=1)


if __name__ == '__main__':
    # preprocess_all()
    deal_all_with_mul()