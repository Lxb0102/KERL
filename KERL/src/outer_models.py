import torch
import torch.nn as nn
import numpy as np
import random
device = 'cuda:0'
from sklearn.metrics import (
    jaccard_score,
    roc_auc_score,
    precision_score,
    f1_score,
    average_precision_score,
    )
from itertools import chain


class GCN(nn.Module):
    def __init__(self,graph,emb_matrix=None,random_strage=2,filter_layers=2,emb_dim=64,id_rate=1,device='cpu'):
        super().__init__()
        self.strage = True
        if not torch.is_tensor(graph):
            self.graph = torch.tensor(graph,device=device)
        else:
            self.graph = graph.to(device)

        self.device=device
        if not torch.is_tensor(emb_matrix):

            self.emb_matrix = torch.randn([graph.size()[0],emb_dim]).to(device)
            self.emb_dim = emb_dim
        else:
            self.emb_matrix = emb_matrix
            self.emb_dim = emb_matrix.size()[-1]
        self.activate = nn.Tanh()
        # inner = torch.sqrt((torch.diag(self.graph.sum(dim=0))))
        ident_matrix = torch.eye(self.graph.size()[0],device=device)*id_rate
        self.emb_matrix.requires_grad = True
        self.prop_layer = ident_matrix + self.graph

        self.prop_layer = self.prop_layer.to(torch.float32)
        self.prop_layer.requires_grad = False
        self.filters = [[nn.Linear(emb_dim,emb_dim,device=device),self.activate] for i in range(filter_layers)]
        self.random_strage = random_strage
        self.layers = self.build_layers()
    def propagate(self,input):
        return self.prop_layer@input
    def filt(self,input,filter):
        return filter[1](filter[0](input))
    def build_layers(self):
        layers = self.filters
        for i in range(self.random_strage):
            layers.append([self.propagate])

        # layers = random.sample(layers, len(layers))
        layers = [layers[2],layers[0]]
        layers = list(chain.from_iterable(layers))
        return layers
    def forward(self):
        out = self.emb_matrix
        for layer in self.layers:
            out = layer(out)
        return out

class trd_encoder(nn.Module):
    def __init__(self,emb_dim=64,out_dim=None,device='cpu'):
        super().__init__()
        if not torch.is_tensor(out_dim):
            out_dim = emb_dim
        #forward_encoder
        self.f_encoder = nn.GRU(emb_dim, out_dim, batch_first=True, device=device,dropout=0.2)
        #reverse_encoder
        # self.r_encoder = nn.GRU(emb_dim, emb_dim, batch_first=True, device=device, dropout=0.1)
        #token_random_drop_encoder
        # self.trd_encoder = nn.GRU(emb_dim,emb_dim, batch_first=True, device=device, dropout=0.1)

    def forward(self,input):

        f_out = self.f_encoder(input)[1].transpose(0,1)
        r_out = self.f_encoder(torch.flip(input,dims=[1]))[1].transpose(0, 1)

        return f_out,r_out#,trd_f_out,trd_r_out

class trd_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,input,positive,negative_1=None,negative_2=None):
        x = torch.exp(-(input-positive)**2/2).sum(dim=-1)

        y = torch.exp(-(input+positive-negative_1-negative_2)**2/2).sum(dim=-1)

        return -torch.log(x).sum()


def ddi_rate_score(record, path=None):
    # ddi rate
    ddi_A = path
    all_cnt = 0
    dd_cnt = 0
    tril_matrix = torch.tril(torch.ones(131,131).to(record[0]))
    tril_matrix -= torch.eye(131,device=device)

    for output in record:
        dd_cnt += (output*ddi_A*tril_matrix).sum().sum()
        all_cnt += (output*tril_matrix).sum().sum()

    if all_cnt == 0:
        return 0,1
    return dd_cnt, all_cnt

def multi_label_metric(y_gt, y_pred, y_prob,voc_size=None):

    y_gt = torch.concat(y_gt).reshape(-1,voc_size[2]).cpu().detach().numpy()
    y_pred = torch.cat(y_pred).cpu().reshape(-1,voc_size[2]).detach().numpy()
    y_prob = torch.cat(y_prob).cpu().reshape(-1,voc_size[2]).detach().numpy()


    def jaccard(y_gt, y_pred):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = np.where(y_pred[b] == 1)[0]
            inter = set(out_list) & set(target)
            union = set(out_list) | set(target)
            jaccard_score = 0 if union == 0 else len(inter) / len(union)
            score.append(jaccard_score)
        return np.mean(score)

    def average_prc(y_gt, y_pred):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = np.where(y_pred[b] == 1)[0]
            inter = set(out_list) & set(target)
            prc_score = 0 if len(out_list) == 0 else len(inter) / len(out_list)
            score.append(prc_score)
        return score


    def f1(y_gt, y_pred):
        all_micro = []
        for b in range(y_gt.shape[0]):
            all_micro.append(f1_score(y_gt[b], y_pred[b], average="macro"))
        return np.mean(all_micro)

    def roc_auc(y_gt, y_prob):
        all_micro = []
        for b in range(len(y_gt)):
            all_micro.append(roc_auc_score(y_gt[b], y_prob[b], average="macro"))
        return np.mean(all_micro)

    def precision_auc(y_gt, y_prob):
        all_micro = []
        for b in range(len(y_gt)):
            all_micro.append(
                average_precision_score(y_gt[b], y_prob[b], average="macro")
            )
        return np.mean(all_micro)

    def precision_at_k(y_gt, y_prob, k=3):
        precision = 0
        sort_index = np.argsort(y_prob, axis=-1)[:, ::-1][:, :k]
        for i in range(len(y_gt)):
            TP = 0
            for j in range(len(sort_index[i])):
                if y_gt[i, sort_index[i, j]] == 1:
                    TP += 1
            precision += TP / len(sort_index[i])
        return precision / len(y_gt)


    prauc = precision_auc(y_gt, y_prob)
    # jaccard
    ja = jaccard(y_gt, y_pred)
    F1 = f1(y_gt,y_pred)

    return ja, prauc ,F1

import torch
import torch.nn as nn

