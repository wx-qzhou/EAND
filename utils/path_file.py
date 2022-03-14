from os.path import abspath, dirname, join, exists
from os import mkdir

import os

def make_dir(path):
    if not exists(path):
        mkdir(path)

file_path = abspath(__file__)
project_path = dirname(dirname(file_path))
# raw data
data_path = join(project_path, "data")
train_raw_data_path = join(data_path, "train")
test_raw_data_path = join(data_path, "test")
model_data_path = join(data_path, "model_none")
make_dir(model_data_path)
# processing
preprocessed_path = join(project_path, "processing_data_new_none") #8, 12, 16, 20, 24, 28, 32 processing_mean_max_6_data C, A, V, K, T, Y
print(preprocessed_path)
make_dir(preprocessed_path)
node_data_path = join(preprocessed_path, "node")
adj_data_path = join(preprocessed_path, "adj")
label_data_path = join(preprocessed_path, "label")
edge_data_path = join(preprocessed_path, "edge")
allfeature_data_path = join(preprocessed_path, "allfeature")
make_dir(preprocessed_path)
make_dir(node_data_path)
make_dir(adj_data_path)
make_dir(label_data_path)
make_dir(edge_data_path)
make_dir(allfeature_data_path)
preprocessed_data_list = os.listdir(adj_data_path)
# IDF of feature
feature_path = join(project_path, "features")
idf_dict_path = join(feature_path, "feature_idf.pkl")
## output
# result_data_path = join(project_path, "output")
# make_dir(result_data_path)

train_raw_data_list = [join(train_raw_data_path, train_raw_data) for train_raw_data in os.listdir(train_raw_data_path)]
test_raw_data_list = [join(test_raw_data_path, train_raw_data) for train_raw_data in os.listdir(test_raw_data_path)]

train_name_list = [os.path.basename(file_name).split(".json")[0].lower() for file_name in train_raw_data_list]
test_name_list = [os.path.basename(file_name).split(".json")[0].lower() for file_name in test_raw_data_list]