import dill
import torch


mimic_ver = 'iii'
if mimic_ver == 'iii':
    data = dill.load(open('records_final.pkl','rb'))
    ccs_voc = dill.load(open(r'diag_proc_ccs.pkl', 'rb'))
    voc = dill.load(open('voc_final.pkl', 'rb'))
else:
    data = dill.load(open('records_final_4.pkl','rb'))
    ccs_voc = dill.load(open(r'diag_proc_ccs_4.pkl', 'rb'))
    voc = dill.load(open('voc_final_4.pkl', 'rb'))

for patient_num in range(len(data)):
        for vst_num in range(len(data[patient_num])):
            diag_ccs_list = [ccs_voc[0]['icd2ccs_idx'][i] for i in data[patient_num][vst_num][0]]
            proc_ccs_list = [ccs_voc[1]['icd2ccs_idx'][i] for i in data[patient_num][vst_num][1]]
            data[patient_num][vst_num].insert(3, list(set(diag_ccs_list)))
            data[patient_num][vst_num].insert(4, list(set(proc_ccs_list)))

diag_voc, pro_voc, med_voc,diag_ccs_voc,proc_ccs_voc, = voc['diag_voc'], \
                                                voc['pro_voc'], voc['med_voc'], ccs_voc[0],ccs_voc[1]

voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word),
                len(diag_ccs_voc['idx2word']), len(proc_ccs_voc['idx2word']))

device = 'cuda'
there_is_ccs = True

diag_labels = torch.zeros(voc_size[0],device=device)
proc_labels = torch.zeros(voc_size[1],device=device)
med_labels = torch.zeros(voc_size[2],device=device)
if there_is_ccs:
    ccs_diag_labels = torch.zeros(voc_size[3], device=device)
    ccs_proc_labels = torch.zeros(voc_size[4], device=device)

diag_co_occur = diag_labels.reshape(-1,1)*diag_labels
proc_co_occur = proc_labels.reshape(-1,1)*proc_labels
if there_is_ccs:
    diag_ccs_icd_co_occur = ccs_diag_labels.reshape(-1,1)*diag_labels
    proc_ccs_icd_co_occur = ccs_proc_labels.reshape(-1, 1) * proc_labels
med_co_occur = med_labels.reshape(-1,1)*med_labels
visit_count = 0
for patient in data:
    for visit in patient:

        diag_labels.fill_(0)
        proc_labels.fill_(0)
        med_labels.fill_(0)

        diag_labels[visit[0]] = 1
        proc_labels[visit[1]] = 1
        med_labels[visit[2]] = 1

        diag_co_occur += diag_labels.reshape(-1, 1) * diag_labels
        proc_co_occur += proc_labels.reshape(-1, 1) * proc_labels
        if there_is_ccs:
            ccs_diag_labels.fill_(0)
            ccs_proc_labels.fill_(0)
            ccs_diag_labels[visit[3]]=1
            ccs_proc_labels[visit[4]] = 1
            diag_ccs_icd_co_occur += ccs_diag_labels.reshape(-1, 1) * diag_labels
            proc_ccs_icd_co_occur += ccs_proc_labels.reshape(-1, 1) * proc_labels
        med_co_occur += med_labels.reshape(-1, 1) * med_labels
        visit_count += 1

diag_co_occur_0_1 = torch.where(diag_co_occur>0,1,0)
proc_co_occur_0_1 = torch.where(proc_co_occur>0,1,0)
med_co_occur_0_1 = torch.where(med_co_occur>0,1,0)
diag_ccs_icd_co_occur_0_1 = torch.where(diag_ccs_icd_co_occur>0,1,0)
proc_ccs_icd_co_occur_0_1 = torch.where(proc_ccs_icd_co_occur>0,1,0)
if there_is_ccs:
    co_occur = {'diag_co_occur':diag_co_occur,
                'proc_co_occur':proc_co_occur,
                'diag_ccs_icd_co_occur':diag_ccs_icd_co_occur,
                'proc_ccs_icd_co_occur':proc_ccs_icd_co_occur,
                'med_co_occur':med_co_occur,
                'visit_count':visit_count}
else:
    co_occur = {'diag_co_occur': diag_co_occur,
                'proc_co_occur': proc_co_occur,
                'med_co_occur': med_co_occur,
                'visit_count': visit_count}

dill.dump(co_occur,open('co_occur_matrix.pkl','wb'))

a=dill.load(open('co_occur_matrix.pkl','rb'))
