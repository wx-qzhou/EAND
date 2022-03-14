from __future__ import division
from __future__ import print_function
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import argparse
import numpy as np
from random import randint

def load_model_param():
    for data in training_generator:
        N, node_feature_list, dense_raw_adj_list, dense_normalized_adj_list, edge_label, class_weight, case_name = \
                    process_loaded_data(data, args.max_graph_size) # N, (1， N， 16), (1，6, 1, N, N), (1,6,1,N,N), (1, N*N), (2), "name" 这里的6是选取的特征数
        return node_feature_list.shape[-1], len(dense_raw_adj_list),

def train(epoch):
    model.train()
    optimizer.zero_grad()
    avg_loss = 0
    total_pair_num = 0
    total_nums = torch.zeros(4, dtype=int)
    total_num = torch.zeros(6, dtype=float)
    for data_index, data in enumerate(training_generator):
        N, node_feature_list, dense_raw_adj_list, dense_normalized_adj_list, edge_label, class_weight, case_name = \
            process_loaded_data(data, args.max_graph_size, randint(0, walk_len))
        output, cos_ = model(node_feature_list, dense_normalized_adj_list, dense_raw_adj_list)
        del node_feature_list
        del dense_normalized_adj_list
        del dense_raw_adj_list
        loss_train = F.nll_loss(output, edge_label.squeeze(), class_weight) + criterion(cos_.squeeze(), edge_label.squeeze())
        preds = output.max(1)[1].type_as(edge_label.squeeze())
        _, _, _, nums = train_utils.evaluate(preds, edge_label.squeeze())
        total_nums += nums
        del data
        del output
        loss_train.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        avg_loss += N * N * float(loss_train)
        total_pair_num += N * N
    avg_loss /= total_pair_num
    tp, fp, fn, tn = total_nums
    precision = tp / (tp + fp + 1e-14)
    recall = tp / (tp + fn + 1e-14)
    f1 = (precision * recall) ** 0.5
    print('train',
          'Epoch: {:04d}'.format(epoch + 1),
          'loss: {:.4f}'.format(avg_loss),
          'precision: {:.4f}'.format(precision),
          'recall: {:.4f}'.format(recall),
          'f1: {:.4f}'.format(f1)
          )

def val(epoch):
    with torch.no_grad():
        model.eval()
        optimizer.zero_grad()
        avg_loss = 0
        total_nums = torch.zeros(4, dtype=int)
        total_num = torch.zeros(6, dtype=float)
        prf_nums = torch.zeros(3, dtype=float)
        total_pair_num = 0
        time = 0
        for data_index, data in enumerate(val_generator):
            N, node_feature_list, dense_raw_adj_list, dense_normalized_adj_list, edge_label, class_weight, case_name = \
                process_loaded_data(data, args.max_graph_size)
            output, cos_ = model(node_feature_list, dense_normalized_adj_list, dense_raw_adj_list)
            del node_feature_list
            del dense_normalized_adj_list
            del dense_raw_adj_list
            loss_val = F.nll_loss(output, edge_label.squeeze(), class_weight) + criterion(cos_.squeeze(), edge_label.squeeze())
            preds = output.max(1)[1].type_as(edge_label.squeeze())
            acc_val, recall_val, f1_val, nums = train_utils.evaluate(preds, edge_label.squeeze())
            avg_loss += float(loss_val) * N * N
            prf_nums += torch.tensor([acc_val, recall_val, f1_val])
            total_nums += nums
            total_pair_num += N * N
            time += 1
            if epoch == -1:
                print(  "{0}:".format(val_case_name[data_index]),
                        'precision: {:.4f}'.format(acc_val),
                        'recall: {:.4f}'.format(recall_val),
                        'f1: {:.4f}'.format(f1_val)
                        )

        avg_loss /= total_pair_num
        tp, fp, fn, tn = total_nums
        precision = tp / (tp + fp + 1e-14)
        recall = tp / (tp + fn + 1e-14)
        f1 = 2 * precision * recall / (precision + recall + 1e-14)
        macro_precision = prf_nums[0] / time
        macro_recall = prf_nums[1] / time
        macro_f1 = 2 * macro_precision * macro_recall / (macro_precision + macro_recall + 1e-14)
        global best_f1
        if best_f1 < macro_f1:
            best_f1 = macro_f1
            print(best_f1)
            torch.save(model.state_dict(), path_file.model_data_path + '/EGNN1.ckp')
        print('val',
              'Epoch: {:04d}'.format(epoch + 1),
              'loss: {:.4f}'.format(avg_loss),
              'precision: {:.4f}'.format(precision),
              'recall: {:.4f}'.format(recall),
              'f1: {:.4f}'.format(f1),
              'tp: {:.4f}'.format(tp),
              'fp: {:.4f}'.format(fp),
              'fn: {:.4f}'.format(fn)
              )
        print(
              'precision_macro_avg: {:.4f}'.format(macro_precision),
              'recall_macro_avg: {:.4f}'.format(macro_recall),
              'f1_macro_avg: {:.4f}'.format(macro_f1)
              )

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden_1', type=int, default=12,
                        help='Number of hidden units of 1st GCN layer')
    parser.add_argument('--hidden_2', type=int, default=12,
                        help='Number of hidden units of 2nd GCN layer')
    parser.add_argument('--hidden_3', type=int, default=12,
                        help='Number of hidden units of 1st FC layer.')
    parser.add_argument('--hidden_4', type=int, default=12,
                        help='Number of hidden units of 2nd FC layer.')
    parser.add_argument('--n_classes', type=int, default=2,
                        help='Number of edge classes')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--device', type=str, default='2,3',
                        help='cuda device number')
    parser.add_argument('--max_graph_size', type=int, default=3500,
                        help='under-sampling graphs whose sizes are bigger than the value')
    parser.add_argument('--state_dict_path', type=str, default=None,
                        help='load and test')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

    import torch
    import torch.nn.functional as F
    import torch.optim as optim

    # Training settings
    from model.EGNNA import EGNNA
    from utils import path_file, train_utils
    from data_loader import Dataset, process_loaded_data
    from torch.utils import data
    walk_len = 9 # 5, 7, 9, 11, 13, 15
    print(walk_len)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    case_name_list = path_file.preprocessed_data_list
    train_case_name = list(set(path_file.train_name_list) & set(path_file.preprocessed_data_list))
    val_case_name = list(set(path_file.test_name_list) & set(path_file.preprocessed_data_list))
    print('Train case num: {}'.format(len(train_case_name)))
    print('Validation case num: {}'.format(len(val_case_name)))

    train_set = Dataset(train_case_name)
    training_generator = data.DataLoader(train_set, batch_size=1, shuffle=True, pin_memory=True) # pin_memory：数据拷贝

    val_set = Dataset(val_case_name)
    val_generator = data.DataLoader(val_set, batch_size=1, shuffle=False, pin_memory=True)
    n_node_features, n_edge_features = load_model_param() # 获取node 的特征数16，edge 的特征数6
    print("n_edge_features", n_edge_features)
    model = EGNNA(n_node_features, # M1
                args.hidden_1, # M2
                args.hidden_2, # M3
                args.hidden_3, # M4
                args.hidden_4, # M5
                args.n_classes, # 2
                n_edge_features, # E
                args.dropout)
    if torch.cuda.device_count() > 1: # 设置多gpu 运行
        model = torch.nn.DataParallel(model)
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay) # 设置优化器

    model.cuda()
    criterion = torch.nn.L1Loss()
    best_f1 = -1 

    # Train model
    for epoch in range(args.epochs):
        train(epoch)
        #if epoch % 5 == 0:
        val(epoch)
    print("Optimization Finished!")
    # torch.save(model.state_dict(), path_file.model_data_path + '/EGNN.ckp')

    model.load_state_dict(torch.load(path_file.model_data_path + '/EGNN1.ckp'))
    model.eval()

    # Testing
    val(-1)
