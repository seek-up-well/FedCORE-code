import numpy as np
from preprocess import *
from SharedNeuMF import NeuMF
from generator import *
from const import *
import torch
import argparse
from loguru import logger
from dataclasses import replace
import os
from unittest import result
os.environ['LOGURU_LEVEL'] = 'INFO'
parser = argparse.ArgumentParser()
parser.add_argument("--dataset",
                    type=str,
                    default='ML100KCF',
                    help="dataset name")
parser.add_argument("--lr",
                    type=float,
                    default=0.1,
                    help="learning rate")
parser.add_argument("--copy", 
                    type=str,
                    default='1',
                    help="copy idnex")
parser.add_argument("--gpu",
                    type=str,
                    default="5",
                    help="gpu card ID")
parser.add_argument("--top",
                    type=str,
                    default="1",
                    help="top per cent ")
parser.add_argument("--itemtype",
                    type=str,
                    default="gmf",
                    help="item_type ")
args = parser.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

manual_seed = 0
random.seed(manual_seed)
np.random.seed(manual_seed)
torch.manual_seed(manual_seed)
LR = args.lr
dataset = args.dataset
copy = args.copy
top = args.top
SaveUV_dir = f"exp2_1_table/{dataset}-copy{copy}-top{top}-lr{LR}"
item_type = args.itemtype


def uviid(U, V, iidfind):
    sum_num = 0
    iidnonset = set(iidfind[:, 0])
    print(len(iidfind))
    sum_numlist = []
    for ii in range(len(iidfind)):
        i = iidfind[ii][1]
        target = torch.cosine_similarity(V[i].reshape(
            1, -1), V[iidfind[ii][2]].reshape(1, -1))

        tempdict = []
        for j in range(len(V)):
            if (j not in iidnonset) and (j != iidfind[0][2]):
                tempdict.append(torch.cosine_similarity(
                    V[j].reshape(1, -1), V[i].reshape(1, -1))[0].tolist())
        tempdict = np.array(tempdict)
        tempdict.sort()

        if target > tempdict[-1]:
            sum_num = sum_num + 1

            sum_numlist.append([target[0][0], ii])
    print(sum_num)
    ''' print(sum_numlist) '''
    print()
    return sum_num


def item_set_change(data1itemuni_set, data2itemuni_set, dataitemcom_set, iidfind):
    for i in range(len(iidfind)):
        dataitemcom_set.remove(iidfind[i][0])
        data1itemuni_set.add(iidfind[i][1])
        data2itemuni_set.add(iidfind[i][2])
    return data1itemuni_set, data2itemuni_set, dataitemcom_set


def cal__max_sim_num(iidfind, model_a, model_b, data2itemuni_set, top_num):
    sum_num = 0
    iidnonset = set(iidfind[:, 0])
    print(len(iidfind))
    sum_numlist = []

    for i in range(len(iidfind)):
        target = torch.cosine_similarity(model_a.state_dict()[f'{item_type}_item_embedding.weight'][iidfind[i][1]], model_b.state_dict()[
                                         f'{item_type}_item_embedding.weight'][iidfind[i][2]], dim=0)
        temp_sim_list = []
        for item_id in data2itemuni_set:
            if iidfind[i][2] != item_id:
                temp_sim_list.append(torch.cosine_similarity(model_a.state_dict()[
                                     f'{item_type}_item_embedding.weight'][iidfind[i][1]], model_b.state_dict()[f'{item_type}_item_embedding.weight'][item_id], dim=0))
        temp_sim_list = np.array(temp_sim_list)
        temp_sim_list.sort()

        if target > temp_sim_list[-top_num]:
            sum_num = sum_num + 1
            sum_numlist.append([target, i])
    print(sum_num)
    ''' print(sum_numlist) '''
    print()
    return sum_num


if __name__ == "__main__":

    partner_list = ['a']
    ep_list = []
    ep_list.append((0, 0))
    ep_list.append((0, 1))
    ep_list.append((0, 5))
    ep_list.append((0, 20))
    ep_list.append((1, 0))
    ep_list.append((5, 0))
    ep_list.append((20, 0))
    ep_list.append((49, 0))

    filename = './datasets/'+dataset+'INDEX/'
    data1useruni_list = np.load(filename+'data1useruni.npy', allow_pickle=True) - \
        np.ones_like(np.load(filename+'data1useruni.npy', allow_pickle=True))
    data2useruni_list = np.load(filename+'data2useruni.npy', allow_pickle=True) - \
        np.ones_like(np.load(filename+'data2useruni.npy', allow_pickle=True))
    datausercom_list = np.load(filename+'datausercom.npy', allow_pickle=True) - \
        np.ones_like(np.load(filename+'datausercom.npy', allow_pickle=True))
    data1itemuni_list = np.load(filename+'data1itemuni.npy', allow_pickle=True) - \
        np.ones_like(np.load(filename+'data1itemuni.npy', allow_pickle=True))
    data2itemuni_list = np.load(filename+'data2itemuni.npy', allow_pickle=True) - \
        np.ones_like(np.load(filename+'data2itemuni.npy', allow_pickle=True))
    dataitemcom_list = np.load(filename+'dataitemcom.npy', allow_pickle=True) - \
        np.ones_like(np.load(filename+'dataitemcom.npy', allow_pickle=True))
    data1useruni_set = set(data1useruni_list)
    data2useruni_set = set(data2useruni_list)
    datausercom_set = set(datausercom_list)
    data1itemuni_set = set(data1itemuni_list)
    data2itemuni_set = set(data2itemuni_list)
    dataitemcom_set = set(dataitemcom_list)

    if dataset == 'ML100KCF':
        maxn = 943
    elif dataset == "ML1MCF":
        maxn = 6040
    else:
        maxn = 5000

    top = float(top)
    result_temp = []
    SaveUV_dir = f'exp2_1_table/{dataset}-copy{copy}-lr{LR}'
    save_five_name = f'exp2_1_table/{dataset}-copy{copy}top{top}-lr{LR}'
    iidfind = np.load(SaveUV_dir + "iidfind.npy")
    maxm = iidfind[-1, 1]
    top_num = round(top * 0.01*len(iidfind))

    data1itemuni_set, data2itemuni_set, dataitemcom_set = item_set_change(
        data1itemuni_set, data2itemuni_set, dataitemcom_set, iidfind)
    for j in range(len(ep_list)):
        SaveUV_dir_tempa = SaveUV_dir + \
            f'-partnera-EP{ep_list[j][0]}-ep{ep_list[j][1]}'
        model_a = NeuMF(maxn, maxm, HIDDEN, DROP)
        model_a.load_state_dict(torch.load(
            SaveUV_dir_tempa, map_location=torch.device('cpu')))
        model_a.eval()

        SaveUV_dir_tempb = SaveUV_dir + \
            f'-partnerb-EP{ep_list[j][0]}-ep{ep_list[j][1]}'
        model_b = NeuMF(maxn, maxm, HIDDEN, DROP)
        model_b.load_state_dict(torch.load(
            SaveUV_dir_tempb, map_location=torch.device('cpu')))
        model_b.eval()

        sum_num = cal__max_sim_num(
            iidfind, model_a, model_b, data2itemuni_set, top_num)
        result_temp.append(sum_num)
    np.save(save_five_name+f'{item_type}table_result', np.array(result_temp))
