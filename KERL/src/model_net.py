import torch
import torch.nn as nn
import torch.nn.functional as F
from src.outer_models import RetNetRelPos as RNP
# from outer_models import MolecularGraphNeuralNetwork
from src.outer_models import GCN
from src.outer_models import trd_encoder
from src.layers import GraphConvolution
torch.manual_seed(1203)

class demo_net(nn.Module):
    def __init__(self,voc_size,emb_dim=64, diag_ccs_icd_co=None, proc_ccs_icd_co=None,
                 device='cuda:0',mimic_ver=None):
        super().__init__()
        self.device = device
        self.emb_dim = emb_dim
        self.med_dim = emb_dim*2
        self.voc_size = voc_size
        self.nhead = 2
        self.diag_ccs_icd_co=diag_ccs_icd_co
        self.proc_ccs_icd_co=proc_ccs_icd_co

        if mimic_ver == 'iii':
            self.d = 241
            self.p = 202
        # MIMIC-IV
        else:
            self.d = 217
            self.p = 200

        self.rnp = RNP(emb_dim=emb_dim*4)
        self.diag_emb = nn.Embedding(voc_size[0],emb_dim,padding_idx=0,device=device)
        self.proc_emb = nn.Embedding(voc_size[1],emb_dim,padding_idx=0,device=device)
        self.ccs_diag_emb = nn.Embedding(voc_size[3], emb_dim, padding_idx=0, device=device)
        self.ccs_proc_emb = nn.Embedding(voc_size[4], emb_dim, padding_idx=0, device=device)
        self.dropout = nn.Dropout(p=0.2)

        self.diag_linear_1 = nn.Sequential(*[nn.Linear(emb_dim,emb_dim*2,device=device),
                                           nn.Tanh(),
                                           nn.Linear(emb_dim*2,emb_dim,device=device),
                                           nn.Dropout(0.3)])
        self.proc_linear_1 = nn.Sequential(*[nn.Linear(emb_dim,emb_dim*2,device=device),
                                           nn.Tanh(),
                                           nn.Linear(emb_dim*2,emb_dim,device=device),
                                           nn.Dropout(0.3)])

        self.final_linear = nn.Sequential(*[nn.Linear(self.med_dim,self.med_dim*4,device=device),
                                           nn.Tanh(),
                                           nn.Linear(self.med_dim*4,self.med_dim*4,device=device),
                                           nn.Tanh(),
                                            nn.Linear(self.med_dim*4, self.med_dim, device=device),
                                           nn.Dropout(0.3)])
        self.d_trans =nn.Linear(self.d,64)
        self.p_trans =nn.Linear(self.p,64)

        self.med_block = nn.Parameter(torch.randn([self.med_dim,voc_size[2]-1],device=device))
        self.his_seq_med = nn.Parameter(torch.randn([self.med_dim,voc_size[2]-1],device=device))

        self.diag_med_block = nn.Parameter(torch.randn([self.emb_dim, voc_size[2] - 1], device=device))
        self.proc_med_block = nn.Parameter(torch.randn([self.emb_dim, voc_size[2] - 1], device=device))

        self.diag_encoder = nn.TransformerEncoderLayer(emb_dim, self.nhead, batch_first=True, dropout=0.2,device=device)
        self.proc_encoder = nn.TransformerEncoderLayer(emb_dim, self.nhead, batch_first=True, dropout=0.2,device=device)

        self.diag_integ = trd_encoder(emb_dim=int(self.med_dim/2),device=device)
        self.proc_integ = trd_encoder(emb_dim=int(self.med_dim/2),device=device)


        self.gender_block = nn.Parameter(torch.eye(voc_size[2]-1,device=device,requires_grad=True))

        self.patient_linear = nn.Sequential(*[nn.Linear(self.med_dim, self.med_dim, device=device),
                                              nn.Tanh()
                                              ])
        self.age_block = nn.Parameter(torch.randn(self.med_dim,voc_size[2],device=device,requires_grad=True))

        self.patient_mem_contact = nn.Sequential(*[nn.Linear(self.med_dim*2,self.med_dim*4,device=device),
                                           nn.Tanh(),
                                           nn.Linear(self.med_dim*4,voc_size[2]-1,device=device),
                                           nn.Tanh(),
                                           nn.Dropout(0.2)])
        self.drug_mem_integ = trd_encoder(emb_dim=int(voc_size[2]-1), device=device)


        if mimic_ver == 'iii':
            self.mlp1= nn.Linear(241,1958).to(self.device)
            self.mlp2 = nn.Linear(202, 1430).to(self.device)
        else:
            self.mlp1 = nn.Linear(217, 1892).to(self.device)
            self.mlp2 = nn.Linear(200, 4798).to(self.device)

        self.dia= self.diag_ccs_icd_co.matmul(self.mlp1.weight).to(self.device)

        self.pro = self.proc_ccs_icd_co.matmul(self.mlp2.weight).to(self.device)

        self.gcn1= GCN(voc_size=voc_size[2], emb_dim=emb_dim, adj=self.dia, device=device)
        self.gcn2 = GCN(voc_size=voc_size[2], emb_dim=emb_dim, adj=self.pro, device=device)

    def history_gate_unit(self, patient_rep, all_vst_drug, contacter, his_fuser=None):

        his_seq_mem = patient_rep[:-1]
        his_seq_mem = torch.cat([torch.zeros_like(patient_rep[0]).unsqueeze(dim=0), his_seq_mem], dim=0)

        his_seq_container = torch.zeros([patient_rep.size()[0],
                                         patient_rep.size()[0],
                                         patient_rep.size()[1] * 2],
                                        device=self.device)

        for i in range((len(patient_rep))):
            for j in range((len(his_seq_mem))):
                if j <= i:
                    his_seq_container[i, j] = torch.concat([patient_rep[i],
                                                            his_seq_mem[j]], dim=-1)
        his_seq_container = contacter(his_seq_container)

        his_seq_filter_mask = torch.tril(torch.ones([patient_rep.size()[0],
                                                     patient_rep.size()[0]], device=self.device)).unsqueeze(dim=-1)

        his_seq_enhance = his_seq_filter_mask * his_seq_container * all_vst_drug
        his_seq_enhance = his_seq_enhance.sum(dim=1)

        return his_seq_enhance.reshape(-1, self.voc_size[2] - 1)

    def encoder(self,diag_seq,proc_seq,diag_mask,proc_mask,ccs_diag_seq,ccs_proc_seq,ccs_diag_mask,ccs_proc_mask, diag_coo, proc_coo):
        # print(ccs_diag_mask.size())
        # print(diag_seq.size())
        max_diag_num = diag_seq.size()[-1]
        max_proc_num = proc_seq.size()[-1]
        ccs_max_diag_num = ccs_diag_seq.size()[-1]
        ccs_max_proc_num = ccs_proc_seq.size()[-1]
        max_visit_num = diag_seq.size()[1]

        batch_size = diag_seq.size()[0]
        diag_seq = self.diag_linear_1(self.diag_emb(diag_seq).view(batch_size * max_visit_num,
                                                                   max_diag_num, self.emb_dim))
        proc_seq = self.proc_linear_1(self.proc_emb(proc_seq).view(batch_size * max_visit_num,
                                                                   max_proc_num, self.emb_dim))
        ccs_diag_seq = self.diag_linear_1(self.ccs_diag_emb(ccs_diag_seq).view(batch_size * max_visit_num,
                                                                   ccs_max_diag_num, self.emb_dim))
        ccs_proc_seq = self.proc_linear_1(self.ccs_proc_emb(ccs_proc_seq).view(batch_size * max_visit_num,
                                                                   ccs_max_proc_num, self.emb_dim))

        # >exp
        # diag_seq = torch.zeros_like(diag_seq)
        # ccs_diag_seq = torch.zeros_like(ccs_diag_seq)
        # <

        d_mask_matrix = diag_mask.view(batch_size * max_visit_num, max_diag_num).unsqueeze(dim=1).unsqueeze(
            dim=1).repeat(1, self.nhead, max_diag_num, 1)  # [batch*seq, nhead, input_length, output_length]
        d_mask_matrix = d_mask_matrix.view(batch_size * max_visit_num * self.nhead, max_diag_num, max_diag_num)
        p_mask_matrix = proc_mask.view(batch_size * max_visit_num, max_proc_num).unsqueeze(dim=1).unsqueeze(
            dim=1).repeat(1, self.nhead, max_proc_num, 1)
        p_mask_matrix = p_mask_matrix.view(batch_size * max_visit_num * self.nhead, max_proc_num, max_proc_num)

        ccs_d_mask_matrix = ccs_diag_mask.view(batch_size * max_visit_num, ccs_max_diag_num).unsqueeze(dim=1).unsqueeze(
            dim=1).repeat(1, self.nhead, ccs_max_diag_num, 1)  # [batch*seq, nhead, input_length, output_length]
        ccs_d_mask_matrix = ccs_d_mask_matrix.view(batch_size * max_visit_num * self.nhead, ccs_max_diag_num, ccs_max_diag_num)
        ccs_p_mask_matrix = ccs_proc_mask.view(batch_size * max_visit_num, ccs_max_proc_num).unsqueeze(dim=1).unsqueeze(
            dim=1).repeat(1, self.nhead, ccs_max_proc_num, 1)
        ccs_p_mask_matrix = ccs_p_mask_matrix.view(batch_size * max_visit_num * self.nhead, ccs_max_proc_num, ccs_max_proc_num)

        diag_seq = self.diag_encoder(diag_seq,src_mask=d_mask_matrix).view(-1, max_diag_num,
                                                                           self.emb_dim)
        proc_seq = self.proc_encoder(proc_seq, src_mask=p_mask_matrix).view(-1, max_proc_num,
                                                                            self.emb_dim)

        ccs_diag_seq = self.diag_encoder(ccs_diag_seq, src_mask=ccs_d_mask_matrix).view(-1, ccs_max_diag_num,
                                                                            self.emb_dim)
        ccs_proc_seq = self.proc_encoder(ccs_proc_seq, src_mask=ccs_p_mask_matrix).view(-1, ccs_max_proc_num,
                                                                            self.emb_dim)

        ccs_diag_rep_1, ccs_diag_rep_2 = self.diag_integ(ccs_diag_seq)
        ccs_diag_rep = ccs_diag_rep_1 + ccs_diag_rep_2
        ccs_proc_rep_1, ccs_proc_rep_2 = self.proc_integ(ccs_proc_seq)
        ccs_proc_rep = ccs_proc_rep_1 + ccs_proc_rep_2
        ccs_patient_rep = torch.concat([ccs_diag_rep, ccs_proc_rep], dim=-1)
        ccs_patient_rep = self.final_linear(ccs_patient_rep)

        d = self.d_trans.weight.matmul(diag_coo)
        p = self.p_trans.weight.matmul(proc_coo)

        diag_rep_1,diag_rep_2 = self.diag_integ(diag_seq)
        diag_rep = diag_rep_1+diag_rep_2
        proc_rep_1, proc_rep_2 = self.proc_integ(proc_seq)
        proc_rep = proc_rep_1 + proc_rep_2

        diag_rep=diag_rep+torch.matmul(ccs_diag_rep,d)
        proc_rep = proc_rep + torch.matmul(ccs_proc_rep, p)
        patient_rep = torch.concat([diag_rep,proc_rep],dim=-1)
        patient_rep = self.final_linear(patient_rep)
        patient_rep= 0.5*patient_rep + 0.5*ccs_patient_rep


        return patient_rep,ccs_patient_rep

    def decoder(self,drug_mem=None,patient_rep=None,ccs_patient_rep=None):

        drug_mem = torch.nn.functional.one_hot(drug_mem.squeeze(dim=0),
                                               num_classes=self.voc_size[2]).sum(dim=-2)[:, 1:].to(torch.float32)

        drug_mem_pad = torch.zeros_like(drug_mem[0]).unsqueeze(dim=0)

        drug_mem = torch.cat([drug_mem_pad, drug_mem], dim=0)[:drug_mem.size()[0]].unsqueeze(dim=0).repeat([
            patient_rep.size()[0], 1, 1]).to(self.device)

        #+++++++++++++++++++++++++++++++++++++++++++++
        patient_rep = patient_rep.squeeze(dim=1)
        his_enhance = self.history_gate_unit(patient_rep, drug_mem, self.patient_mem_contact)
        #============================================================
        # +++++++++++++++++++++++++++++++++++++++++++++
        ccs_patient_rep = ccs_patient_rep.squeeze(dim=1)
        ccs_his_enhance = self.history_gate_unit(ccs_patient_rep, drug_mem, self.patient_mem_contact)
        # ============================================================

        final_prob_1 = patient_rep@self.med_block
        ccs_final_prob_1 = ccs_patient_rep@self.med_block

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++

        prob = final_prob_1 + his_enhance
        ccs_prob = ccs_final_prob_1 + ccs_his_enhance
        prob = prob.reshape(-1,self.voc_size[2]-1)
        ccs_prob = ccs_prob.reshape(-1, self.voc_size[2] - 1)
        prob_padder = torch.full_like(prob.T[0], 0).unsqueeze(dim=0).T
        #===visable===
        vs_prob = torch.cat([prob_padder,F.sigmoid(prob)],dim=-1)
        vs_ccs_prob = torch.cat([prob_padder,F.sigmoid(ccs_prob)],dim=-1)
        #=============
        prob = prob + ccs_prob
        prob = F.sigmoid(prob)
        prob = torch.cat([prob_padder, prob], dim=-1)
        # print(vs_prob.size())
        # print(prob.size())
        return prob,prob*prob.T.unsqueeze(dim=-1),vs_prob,vs_ccs_prob
               #[diag_prob_1,diag_prob_2],\
               #[proc_prob_1,proc_prob_2]
               # F.sigmoid(prob_patient_integ_out),\
               # F.sigmoid(his_seq_out)

    def forward(self,input,diag_mask=None,proc_mask=None,ages=None,gender=None,drug_mem=None):
        diag_coo = self.gcn1(adj=self.dia)
        proc_coo = self.gcn2(adj=self.pro)
        patient_rep,ccs_patient_rep = self.encoder(input[0],input[1],input[3],input[4],
                                                     input[5],input[6],input[7],input[8],diag_coo,proc_coo)

        decoder_out = self.decoder(drug_mem=input[2],patient_rep=patient_rep,ccs_patient_rep = ccs_patient_rep)

        return decoder_out


class GCN(nn.Module):
    # self.ehr_gcn = GCN(voc_size=vocab_size[2], emb_dim=emb_dim, adj=ehr_adj, device=device)
    def __init__(self, voc_size, emb_dim, adj, device=torch.device('cpu:0')):
        super(GCN, self).__init__()
        self.voc_size = voc_size
        self.emb_dim = emb_dim
        self.device = device
        self.w = adj.shape[0]

        adj = adj + torch.eye(self.w).to('cuda:0')

        self.x = torch.eye(self.w).to(device)

        self.gcn1 = GraphConvolution(self.w, emb_dim)
        self.dropout = nn.Dropout(p=0.3)
        self.gcn2 = GraphConvolution(emb_dim, 64)  ## graph_transformer 改为16

    def forward(self,adj):
        node_embedding = self.gcn1(self.x, adj)
        node_embedding = F.relu(node_embedding)
        node_embedding = self.dropout(node_embedding)
        node_embedding = self.gcn2(node_embedding, adj)
        edge = adj

        return node_embedding

