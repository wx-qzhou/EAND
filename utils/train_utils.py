import torch


def evaluate(preds, labels, pos_weight=1):
    labels = labels.squeeze().reshape(-1, 1).squeeze()
    # print(preds)
    zes = torch.Tensor(torch.zeros(labels.shape[0])).type(torch.LongTensor).cuda()
    ons = torch.Tensor(torch.ones(labels.shape[0])).type(torch.LongTensor).cuda()
    tp = int(((preds >= 0.1) & (labels == ons)).sum())
    fp = int(((preds >= 0.1) & (labels == zes)).sum())
    fn = int(((preds < 0.1) & (labels == ons)).sum())
    tn = int(((preds < 0.1) & (labels == zes)).sum())
    tp = int(pos_weight * tp)
    fn = int(pos_weight * fn)
    
    epsilon = 1e-7
    acc = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    return acc, recall, (2 * acc * recall) / (acc + recall + 1e-13), torch.tensor([tp, fp, fn, tn])


def cuda_list_object_1d(list_1d, device=None, non_blocking=False):
    return [e.cuda(device, non_blocking) for e in list_1d]
