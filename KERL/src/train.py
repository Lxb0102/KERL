
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from src.data_loader import pad_batch_v2_train, pad_batch_v2_eval
import numpy as np
from src.util import llprint
import dill
from src.outer_models import multi_label_metric
from src.model_net import demo_net
torch.manual_seed(1203)
import random

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def get_items(dataloader, *index):
    sorted_indices = [idx for idx, _ in sorted(enumerate(index), key=lambda x: x[1])]
    index = sorted(index)
    container = [0]*len(index)
    output = []
    for batch_index, batch in enumerate(dataloader):
        while True:
            if batch_index==index:
                container[sorted_indices[0]]=batch
                index = index[1:]
                sorted_indices = sorted_indices[1:]
            else:
                break
    return output


def compute_kl_loss(p, q, pad_mask = None):

    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')

    # pad_mask is for seq-level tasks
    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)

    # You can choose whether to use function "sum" and "mean" depending on your task
    p_loss = p_loss.sum()
    q_loss = q_loss.sum()

    loss = (p_loss + q_loss) / 2
    return loss


def main(temp_number=None):
    device = "cuda"
    mimic_ver = 'iv'
    voc = dill.load(open(r'datas\voc_final_{}.pkl'.format(mimic_ver), 'rb'))
    ccs_voc = dill.load(open(r'datas\diag_proc_ccs_{}.pkl'.format(mimic_ver), 'rb'))
    data = dill.load(open(r'datas\records_final_{}.pkl'.format(mimic_ver), 'rb'))#[::100]
    co_occur = dill.load(open('datas\co_occur_matrix_{}.pkl'.format(mimic_ver), 'rb'))
    diag_ccs_icd_co = F.normalize(co_occur['diag_ccs_icd_co_occur'],dim=-1)
    proc_ccs_icd_co = F.normalize(co_occur['proc_ccs_icd_co_occur'],dim=-1)

    for patient_num in range(len(data)):
        for vst_num in range(len(data[patient_num])):
            diag_ccs_list = [ccs_voc[0]['icd2ccs_idx'][i] for i in data[patient_num][vst_num][0]]
            proc_ccs_list = [ccs_voc[1]['icd2ccs_idx'][i] for i in data[patient_num][vst_num][1]]
            data[patient_num][vst_num].insert(3, list(set(diag_ccs_list)))
            data[patient_num][vst_num].insert(4, list(set(proc_ccs_list)))


    for patient in range(len(data)):
        for vst in range(len(data[patient])):
            # print(data[patient][vst][0])
            data[patient][vst][0]=[i+1 for i in data[patient][vst][0]]
            data[patient][vst][1]=[i+1 for i in data[patient][vst][1]]
            data[patient][vst][2]=[i+1 for i in data[patient][vst][2]]
            data[patient][vst][3] = [i + 1 for i in data[patient][vst][3]]
            data[patient][vst][4] = [i + 1 for i in data[patient][vst][4]]
    # print(data)


    diag_voc, pro_voc, med_voc,diag_ccs_voc,proc_ccs_voc, = voc['diag_voc'], \
                                                            voc['pro_voc'], voc['med_voc'], ccs_voc[0],ccs_voc[1]

    voc_size = (len(diag_voc.idx2word)+1, len(pro_voc.idx2word)+1, len(med_voc.idx2word)+1,
                len(diag_ccs_voc['idx2word'])+1, len(proc_ccs_voc['idx2word'])+1)

    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]

    eval_len = int(len(data[split_point:]) / 2)
    data_eval = data[split_point + eval_len:]
    data_test = data[split_point:split_point + eval_len]

    #============================
    print(voc_size)
    # return None
    #=============train_data_load===============
    train_loader = DataLoader([[j[:3] for j in i] for i in data_train], batch_size=1, collate_fn=pad_batch_v2_train, shuffle=False, pin_memory=False)
    ccs_train_loader = DataLoader([[[j[3],j[4],j[2]] for j in i] for i in data_train], batch_size=1, collate_fn=pad_batch_v2_train, shuffle=False, pin_memory=False)

    train_loader = [patient[:5] for patient in train_loader]
    ccs_train_loader = [patient[:2]+patient[3:] for patient in ccs_train_loader]

    train_loader = [train_loader[patient]+ccs_train_loader[patient]
                    for patient in range(len(train_loader))]

    eval_loader = DataLoader([[j[:3] for j in i] for i in data_eval], batch_size=1, collate_fn=pad_batch_v2_eval,
                              shuffle=False, pin_memory=False)
    ccs_eval_loader = DataLoader([[[j[3], j[4], j[2]] for j in i] for i in data_eval], batch_size=1,
                                  collate_fn=pad_batch_v2_eval, shuffle=False, pin_memory=False)
    eval_loader = [patient[:5] for patient in eval_loader]
    ccs_eval_loader = [patient[:2] + patient[3:] for patient in ccs_eval_loader]
    eval_loader = [eval_loader[patient]+ccs_eval_loader[patient]
                    for patient in range(len(eval_loader))]

    model = demo_net(emb_dim=64, voc_size=voc_size, diag_ccs_icd_co=diag_ccs_icd_co, proc_ccs_icd_co=proc_ccs_icd_co, device=device, mimic_ver=mimic_ver ).to(device)

    print('parameters', get_n_params(model))
    optimizer = Adam(model.parameters(), lr=0.0001)
    EPOCH = 50
    demo_loss_1 = nn.BCELoss()

    for epoch in range(EPOCH):
        ddi_rate = 0
        avg_precise = 0
        avg_recall = 0
        avg_f1 = 0
        count = 1e-6
        model.to(device)
        model.train()
        model_train = True
        all_his = True

        if model_train:
            for index,datas in enumerate(train_loader):

                datas = [i.to(device) for i in datas]
                output = list(model(datas))

                gt_container = torch.zeros_like(output[0], device=device).reshape(-1,voc_size[2])
                loss3_target = np.full((output[0].size()), -1).reshape([-1,voc_size[2]])

                #gamenet
                temp_drug_label = []
                if all_his:
                    for batch_idx, batch in enumerate(temp_drug_label):
                        for idx, seq in enumerate(batch.reshape(-1,batch.size()[-1])):
                            for seq_idx,item in enumerate(seq):
                                try:
                                    # print(temp_drug_label.size())
                                    loss3_target[batch_idx][seq_idx] = item
                                except:
                                    print(temp_drug_label.size())
                                    loss3_target[batch_idx][seq_idx] = item
                else:
                    for batch_idx, batch in enumerate(temp_drug_label):
                        for idx, seq in enumerate(batch.reshape(-1,batch.size()[-1])):
                            for seq_idx,item in enumerate(seq):
                                try:
                                    # print(temp_drug_label.size())
                                    loss3_target[batch_idx][seq_idx] = item
                                except:
                                    print(temp_drug_label.size())
                                    loss3_target[batch_idx][seq_idx] = item


                loss3_target = loss3_target.reshape(output[0].size())
                if all_his:
                    for batch_idx, batch in enumerate(datas[2][0]):
                        for idx, seq in enumerate(batch):
                            gt_container[batch_idx][seq] = 1.0
                else:
                    gt_container[0][datas[2][0][-1]] = 1


                gt_container = gt_container.reshape(output[0].size())
                if all_his:
                    gt_container[:,0] = 0
                else:
                    gt_container[0] = 0

                kl_loss = compute_kl_loss(output[0], output[2])

                loss_1 = demo_loss_1(output[0],gt_container)

                loss_2 = F.multilabel_margin_loss(output[0], torch.LongTensor(loss3_target).to(device))


                loss = 0.5 * (loss_1 + 0.02 * loss_2) +  kl_loss

                loss.backward(retain_graph=True)
                optimizer.step()
                optimizer.zero_grad()
                llprint('\r|'+\
                         '#'*int(50*(index/len(train_loader)))+\
                         '-'*int(50-50*(index/len(train_loader)))+\
                         '|{:.2f}%|train_step:{}/{}'.format(100*(index/len(train_loader)),index,len(train_loader))
                        )

        print()
        model.eval()
        prob_container = []
        gt_container = []
        labels_container = []
        ddi_cnt = 0
        ddi_all_cnt = 0
        avg_med = 0
        for index, datas in enumerate(eval_loader):
            datas = [i.to(device) for i in datas]
            output,_,output_2,output_3 = model(datas)

            gt_data = datas[2][0]

            for idx,vst in enumerate(output.reshape(-1,voc_size[2])):
                gt_temp = torch.zeros_like(vst, device=device)
                if all_his:
                    gt_temp[gt_data[idx]] = 1
                else:
                    gt_temp[gt_data[-1]] = 1
                gt_temp[0] = 0
                avg_med += vst.sum()
                out_labels = torch.where(vst > 0.35, 1.0, 0.0)

                out_numbers = torch.nonzero(out_labels.squeeze())
                ddi_temp_container = out_labels*out_labels.T.unsqueeze(dim=-1)
                labels_container.append(out_labels)
                prob_container.append(vst)
                gt_container.append(gt_temp)

                if gt_temp.sum()!=0:
                    precise = (out_labels * gt_temp).sum() / (out_labels.sum() + 1e-9)
                    recall = (out_labels * gt_temp).sum() / (gt_temp.sum() + 1e-9)
                else:
                    continue
                avg_precise += precise
                avg_recall += recall

                if (precise + recall) == 0:
                    continue
                else:
                    f1 = (2.0 * precise * recall) / (precise + recall)
                avg_f1 += f1

                count += 1

            llprint('\r|' + \
                    '@' * int(50 * (index / len(eval_loader))) + \
                    '-' * int(50 - 50 * (index / len(eval_loader))) + \
                    '|{:.2f}%|eval_step:{}/{}'.format(100 * (index / len(eval_loader)), index, len(eval_loader))
                    )
        avg_precise=avg_precise/count
        avg_recall=avg_recall/count
        avg_f1=avg_f1/count

        jac,prauc,F_1 = multi_label_metric(gt_container,labels_container,prob_container,voc_size=voc_size)
        try:
            ddi_rate = ddi_cnt/ddi_all_cnt
        except:
            pass
            ddi_rate = 0

        print('\navg_prc = {}\n'.format(avg_precise),
              'avg_rec = {}\n'.format(avg_recall),
              'jac = {}\n'.format(jac),
              'prauc = {}\n'.format(prauc),
              'avg_f1 = {}\n'.format(avg_f1),
              'ddi_rate = {}\n'.format(ddi_rate),
              'avg_med = {}\n'.format(avg_med/count)
               )

        print(f'epoch{epoch}\n')


for i in range(1):
    main(i)
