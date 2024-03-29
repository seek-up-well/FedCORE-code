

import copy
from cmath import isnan
import random
import numpy as np
from loguru import logger
import torch
from torch.autograd import Variable

from const import *
from generator import *
from SharedNeuMF_attack import NeuMF_attack
from SharedNeuMF import NeuMF
from preprocess import *
from attack_function import *
manual_seed = 0
random.seed(manual_seed)
np.random.seed(manual_seed)
torch.manual_seed(manual_seed)
 

def attack_direct(uid_rating_dict, model_a, model_b, atrain_data_dict, maxn, maxm):

    optimizer = torch.optim.Adam(model_a.parameters(), lr=0.0005)
    model_a.train()
    avg_sim_arrack_origin = 0
    avg_sim_correspond_origin = 0
    avg_sim_arrack_correspond = 0
    avg_sim_arrack_origin_std = []
    avg_sim_correspond_origin_std = []
    avg_sim_arrack_correspond_std = []
    num_count = 0
    data_label_to_test = []
    data_pred_to_test = []

    avg_sim_arrack_origin_ncf = 0
    avg_sim_correspond_origin_ncf = 0
    avg_sim_arrack_correspond_ncf = 0
    avg_sim_arrack_origin_std_ncf = []
    avg_sim_correspond_origin_std_ncf = []
    avg_sim_arrack_correspond_std_ncf = []

    data_label_to_test_ncf = []
    data_pred_to_test_ncf = []

    model_a_temp = NeuMF_attack(maxn, maxm, HIDDEN, DROP)
    for name, param in model_a.state_dict().items():

        if (name != "gmf_item_embedding.weight") and (name != "ncf_item_embedding.weight"):

            model_a_temp.state_dict()[name].copy_(model_a.state_dict()[name])

    model_a_temp.to(DEVICE)
    model_a_temp.train()

    user_num = 0
    for ud in uid_rating_dict:

        user_num = user_num+1
        for vd in range(len(uid_rating_dict[ud])):

            rating = atrain_data_dict[(ud, uid_rating_dict[ud][vd])]

            true_label = rating

            vid_list = uid_rating_dict[ud]

            uid = np.expand_dims(np.array([ud], 'int32'), axis=1)
            vid = np.expand_dims(np.array([vid_list[vd]], 'int32'), axis=1)

            uid = torch.LongTensor(uid).squeeze(1).to(DEVICE)

            vid = torch.LongTensor(vid).squeeze(1).to(DEVICE)
            ratings = torch.Tensor(np.array([rating])).to(DEVICE)

            gmf_item_latent = model_a.state_dict(
            )["gmf_item_embedding.weight"][vid].to(DEVICE)
            ncf_item_latent = model_a.state_dict(
            )["ncf_item_embedding.weight"][vid].to(DEVICE)
            gmf_item_latent.requires_grad = False
            ncf_item_latent.requires_grad = False

            preds = model_a_temp(uid, gmf_item_latent, ncf_item_latent)

            mse_loss = torch.nn.MSELoss()
            loss = mse_loss(preds, ratings)

            dy_dx = torch.autograd.grad(loss, model_a_temp.parameters())
            original_dy_dx = list((_.detach().clone() for _ in dy_dx))

            model_attack = NeuMF_attack(maxn, maxm, HIDDEN, DROP)
            model_attack.load_state_dict(
                copy.deepcopy(model_a_temp.state_dict()))
            model_attack.to(DEVICE)
            model_attack.train()

            attack_y = Variable(torch.rand([1])*4+1)

            attack_y = attack_y.to(DEVICE)
            attack_y.requires_grad = True

            gmf_item_latent_attack = Variable(torch.rand([1, 16])).to(DEVICE)
            ncf_item_latent_attack = Variable(torch.rand([1, 16])).to(DEVICE)
            gmf_item_latent_attack.requires_grad = True
            ncf_item_latent_attack.requires_grad = True

            criterion_attack = torch.nn.MSELoss()

            optimizer_attack = torch.optim.LBFGS(
                [ncf_item_latent_attack, gmf_item_latent_attack, attack_y])

            gmf_target_attackembedding = torch.zeros([16]).to(DEVICE)
            ncf_target_attackembedding = torch.zeros([16]).to(DEVICE)

            num_epoches = 300
            for epoch in range(num_epoches):
                gmf_target_attackembedding = copy.deepcopy(
                    gmf_item_latent_attack.detach().squeeze())
                ncf_target_attackembedding = copy.deepcopy(
                    ncf_item_latent_attack.detach().squeeze())

                def closure():
                    optimizer_attack.zero_grad()

                    out = model_attack(
                        uid, gmf_item_latent_attack, ncf_item_latent_attack)

                    loss_attack = criterion_attack(out, attack_y)

                    grad_attack = torch.autograd.grad(
                        loss_attack, model_attack.parameters(), create_graph=True)

                    grad_diff = 0
                    for gx, gy in zip(grad_attack, original_dy_dx):
                        grad_diff += ((gx - gy) ** 2).sum()
                    grad_diff.backward()

                    return grad_diff

                optimizer_attack.step(closure)
                if ((torch.isnan(gmf_item_latent_attack.detach().squeeze()).any()) == True) or (torch.isnan(ncf_item_latent_attack.detach().squeeze()).any() == True) or ((torch.isinf(gmf_item_latent_attack.detach().squeeze()).any()) == True) or (torch.isinf(ncf_item_latent_attack.detach().squeeze()).any() == True):
                    gmf_item_latent_attack = copy.deepcopy(
                        gmf_target_attackembedding)
                    ncf_item_latent_attack = copy.deepcopy(
                        ncf_target_attackembedding)

                    gmf_item_latent_attack = gmf_item_latent_attack.unsqueeze(
                        0)
                    ncf_item_latent_attack = ncf_item_latent_attack.unsqueeze(
                        0)
                    break

            model_attack.eval()

            arrack_embedding = gmf_target_attackembedding
            origin_embedding = model_a.state_dict(
            )['gmf_item_embedding.weight'][vid].squeeze()
            correspond_b_embedding = model_b.state_dict(
            )['gmf_item_embedding.weight'][vid-1].squeeze()

            arrack_embedding_ncf = ncf_target_attackembedding
            origin_embedding_ncf = model_a.state_dict(
            )['ncf_item_embedding.weight'][vid].squeeze()
            correspond_b_embedding_ncf = model_b.state_dict(
            )['ncf_item_embedding.weight'][vid-1].squeeze()

            avg_sim_arrack_origin = avg_sim_arrack_origin + \
                torch.cosine_similarity(
                    arrack_embedding, origin_embedding, dim=0).cpu().numpy()
            avg_sim_correspond_origin = avg_sim_correspond_origin + \
                torch.cosine_similarity(
                    arrack_embedding, correspond_b_embedding, dim=0).cpu().numpy()
            avg_sim_arrack_correspond = avg_sim_arrack_correspond + \
                torch.cosine_similarity(
                    origin_embedding, correspond_b_embedding, dim=0).cpu().numpy()
            print("final:")
            print(torch.cosine_similarity(arrack_embedding,
                  origin_embedding, dim=0).cpu().numpy())

            avg_sim_arrack_origin_std.append(torch.cosine_similarity(
                arrack_embedding, origin_embedding, dim=0).cpu().numpy())
            avg_sim_correspond_origin_std.append(torch.cosine_similarity(
                arrack_embedding, correspond_b_embedding, dim=0).cpu().numpy())
            avg_sim_arrack_correspond_std.append(torch.cosine_similarity(
                origin_embedding, correspond_b_embedding, dim=0).cpu().numpy())
            num_count = num_count + 1

            avg_sim_arrack_origin_ncf = avg_sim_arrack_origin_ncf + \
                torch.cosine_similarity(
                    arrack_embedding_ncf, origin_embedding_ncf, dim=0).cpu().numpy()
            avg_sim_correspond_origin_ncf = avg_sim_correspond_origin_ncf + \
                torch.cosine_similarity(
                    arrack_embedding_ncf, correspond_b_embedding_ncf, dim=0).cpu().numpy()
            avg_sim_arrack_correspond_ncf = avg_sim_arrack_correspond_ncf + \
                torch.cosine_similarity(
                    origin_embedding_ncf, correspond_b_embedding_ncf, dim=0).cpu().numpy()
            avg_sim_arrack_origin_std_ncf.append(torch.cosine_similarity(
                arrack_embedding_ncf, origin_embedding_ncf, dim=0).cpu().numpy())
            avg_sim_correspond_origin_std_ncf.append(torch.cosine_similarity(
                arrack_embedding_ncf, correspond_b_embedding_ncf, dim=0).cpu().numpy())
            avg_sim_arrack_correspond_std_ncf.append(torch.cosine_similarity(
                origin_embedding_ncf, correspond_b_embedding_ncf, dim=0).cpu().numpy())

            attack_rating = list(model_a_temp(
                uid, gmf_item_latent_attack, ncf_item_latent_attack).detach().cpu().numpy())[0]
            data_label_to_test.append(rating)
            data_pred_to_test.append(attack_rating)

            attack_rating_ncf = list(model_a_temp(
                uid, gmf_item_latent_attack, ncf_item_latent_attack).detach().cpu().numpy())[0]
            data_label_to_test_ncf.append(rating)
            data_pred_to_test_ncf.append(attack_rating_ncf)

    avg_sim_arrack_origin_std = np.array(avg_sim_arrack_origin_std)
    avg_sim_correspond_origin_std = np.array(avg_sim_correspond_origin_std)
    avg_sim_arrack_correspond_std = np.array(avg_sim_arrack_correspond_std)
    std1 = np.std(avg_sim_arrack_origin_std)
    std2 = np.std(avg_sim_correspond_origin_std)
    std3 = np.std(avg_sim_arrack_correspond_std)
    MAE, RMSE = test_acc(data_label_to_test, data_pred_to_test)
    avg_sim_arrack_origin = avg_sim_arrack_origin / num_count
    avg_sim_correspond_origin = avg_sim_correspond_origin / num_count
    avg_sim_arrack_correspond = avg_sim_arrack_correspond / num_count

    avg_sim_arrack_origin_std_ncf = np.array(avg_sim_arrack_origin_std_ncf)
    avg_sim_correspond_origin_std_ncf = np.array(
        avg_sim_correspond_origin_std_ncf)
    avg_sim_arrack_correspond_std_ncf = np.array(
        avg_sim_arrack_correspond_std_ncf)
    std1_ncf = np.std(avg_sim_arrack_origin_std_ncf)
    std2_ncf = np.std(avg_sim_correspond_origin_std_ncf)
    std3_ncf = np.std(avg_sim_arrack_correspond_std_ncf)
    MAE_ncf, RMSE_ncf = test_acc(data_label_to_test_ncf, data_pred_to_test_ncf)
    avg_sim_arrack_origin_ncf = avg_sim_arrack_origin_ncf / num_count
    avg_sim_correspond_origin_ncf = avg_sim_correspond_origin_ncf / num_count
    avg_sim_arrack_correspond_ncf = avg_sim_arrack_correspond_ncf / num_count

    return avg_sim_arrack_origin, avg_sim_correspond_origin, avg_sim_arrack_correspond, MAE, RMSE, std1, std2, std3, avg_sim_arrack_origin_ncf, avg_sim_correspond_origin_ncf, avg_sim_arrack_correspond_ncf, MAE_ncf, RMSE_ncf, std1_ncf, std2_ncf, std3_ncf


def attack_average(uid_rating_dict, model_a, model_b, atrain_data_dict, maxn, maxm):
    optimizer = torch.optim.Adam(model_a.parameters(), lr=0.0005)
    model_a.train()
    avg_sim_arrack_origin = 0
    avg_sim_correspond_origin = 0
    avg_sim_arrack_correspond = 0
    avg_sim_arrack_origin_std = []
    avg_sim_correspond_origin_std = []
    avg_sim_arrack_correspond_std = []
    num_count = 0
    data_label_to_test = []
    data_pred_to_test = []

    avg_sim_arrack_origin_ncf = 0
    avg_sim_correspond_origin_ncf = 0
    avg_sim_arrack_correspond_ncf = 0
    avg_sim_arrack_origin_std_ncf = []
    avg_sim_correspond_origin_std_ncf = []
    avg_sim_arrack_correspond_std_ncf = []

    data_label_to_test_ncf = []
    data_pred_to_test_ncf = []

    model_a_temp = NeuMF_attack(maxn, maxm, HIDDEN, DROP)
    for name, param in model_a.state_dict().items():

        if (name != "gmf_item_embedding.weight") and (name != "ncf_item_embedding.weight"):

            model_a_temp.state_dict()[name].copy_(model_a.state_dict()[name])

    model_a_temp.to(DEVICE)
    model_a_temp.train()

    user_num = 0
    for ud in uid_rating_dict:
        print('user_num', user_num)
        user_num = user_num+1

        uid = torch.LongTensor(np.expand_dims(
            np.array([ud], 'int32'), axis=1)).to(DEVICE)
        uids = torch.LongTensor(np.expand_dims(
            np.array([ud]*5, 'int32'), axis=1)).squeeze(1).to(DEVICE)
        vids = torch.LongTensor(np.expand_dims(
            np.array(uid_rating_dict[ud], 'int32'), axis=1)).squeeze(1).to(DEVICE)
        ratings_list = []
        for vd in range(len(uid_rating_dict[ud])):
            ratings_list.append(
                atrain_data_dict[(ud, uid_rating_dict[ud][vd])])
        true_labels = ratings_list

        ratings = torch.Tensor(np.expand_dims(
            np.array(ratings_list, 'int32'), axis=1)).squeeze(1).to(DEVICE)

        gmf_item_latent = model_a.state_dict(
        )["gmf_item_embedding.weight"][vids].to(DEVICE)
        ncf_item_latent = model_a.state_dict(
        )["ncf_item_embedding.weight"][vids].to(DEVICE)
        gmf_item_latent.requires_grad = False
        ncf_item_latent.requires_grad = False

        preds = model_a_temp(uids, gmf_item_latent, ncf_item_latent)

        mse_loss = torch.nn.MSELoss()
        loss = mse_loss(preds, ratings)

        dy_dx = torch.autograd.grad(loss, model_a_temp.parameters())
        original_dy_dx = list((_.detach().clone() for _ in dy_dx))

        model_attack = NeuMF_attack(maxn, maxm, HIDDEN, DROP)
        model_attack.load_state_dict(copy.deepcopy(model_a_temp.state_dict()))
        model_attack.to(DEVICE)
        model_attack.train()

        attack_y = Variable(torch.rand([5])*4+1)

        attack_y = attack_y.to(DEVICE)
        attack_y.requires_grad = True

        gmf_item_latent_attack = Variable(torch.rand([5, 16])).to(DEVICE)
        ncf_item_latent_attack = Variable(torch.rand([5, 16])).to(DEVICE)
        gmf_item_latent_attack.requires_grad = True
        ncf_item_latent_attack.requires_grad = True

        criterion_attack = torch.nn.MSELoss()

        optimizer_attack = torch.optim.LBFGS(
            [ncf_item_latent_attack, gmf_item_latent_attack, attack_y])
        gmf_target_attackembedding = torch.zeros([5, 16]).to(DEVICE)
        ncf_target_attackembedding = torch.zeros([5, 16]).to(DEVICE)

        num_epoches = 300
        for epoch in range(num_epoches):
            gmf_target_attackembedding = copy.deepcopy(
                gmf_item_latent_attack.detach().squeeze())
            ncf_target_attackembedding = copy.deepcopy(
                ncf_item_latent_attack.detach().squeeze())

            def closure():

                optimizer_attack.zero_grad()

                out = model_attack(
                    uids, gmf_item_latent_attack, ncf_item_latent_attack)

                loss_attack = criterion_attack(out, attack_y)

                grad_attack = torch.autograd.grad(
                    loss_attack, model_attack.parameters(), create_graph=True)

                grad_diff = 0
                for gx, gy in zip(grad_attack, original_dy_dx):
                    grad_diff += ((gx - gy) ** 2).sum()
                grad_diff.backward()

                if (epoch) % 100 == 0:
                    print("Epoch[{}/{}], loss: {:.6f}".format(epoch,
                          num_epoches, loss_attack.data))
                    model_attack.eval()

                    arrack_embedding = gmf_item_latent_attack.detach().squeeze()
                    origin_embedding = model_a.state_dict(
                    )['gmf_item_embedding.weight'][vids].squeeze()
                    correspond_b_embedding = model_b.state_dict(
                    )['gmf_item_embedding.weight'][vids-torch.LongTensor(np.array([1]*5)).to(DEVICE)].squeeze()

                    print(torch.cosine_similarity(arrack_embedding,
                          origin_embedding, dim=1).cpu().numpy())
                    print(torch.cosine_similarity(arrack_embedding,
                          correspond_b_embedding, dim=1).cpu().numpy())
                    print(torch.cosine_similarity(origin_embedding,
                          correspond_b_embedding, dim=1).cpu().numpy())
                    model_attack.train()
                return grad_diff

            optimizer_attack.step(closure)
            if ((torch.isnan(gmf_item_latent_attack.detach().squeeze()).any()) == True) or (torch.isnan(ncf_item_latent_attack.detach().squeeze()).any() == True) or ((torch.isinf(gmf_item_latent_attack.detach().squeeze()).any()) == True) or (torch.isinf(ncf_item_latent_attack.detach().squeeze()).any() == True):
                gmf_item_latent_attack = copy.deepcopy(
                    gmf_target_attackembedding)
                ncf_item_latent_attack = copy.deepcopy(
                    ncf_target_attackembedding)

                break

        print()

        model_attack.eval()

        arrack_embedding = gmf_target_attackembedding
        origin_embedding = model_a.state_dict(
        )['gmf_item_embedding.weight'][vids].squeeze()
        correspond_b_embedding = model_b.state_dict(
        )['gmf_item_embedding.weight'][vids-torch.LongTensor(np.array([1]*5)).to(DEVICE)].squeeze()

        arrack_embedding_ncf = ncf_target_attackembedding
        origin_embedding_ncf = model_a.state_dict(
        )['ncf_item_embedding.weight'][vids].squeeze()
        correspond_b_embedding_ncf = model_b.state_dict(
        )['ncf_item_embedding.weight'][vids-torch.LongTensor(np.array([1]*5)).to(DEVICE)].squeeze()

        avg_sim_arrack_origin = avg_sim_arrack_origin + \
            np.sum(torch.cosine_similarity(arrack_embedding,
                   origin_embedding, dim=1).cpu().numpy())
        avg_sim_correspond_origin = avg_sim_correspond_origin + \
            np.sum(torch.cosine_similarity(arrack_embedding,
                   correspond_b_embedding, dim=1).cpu().numpy())
        avg_sim_arrack_correspond = avg_sim_arrack_correspond + \
            np.sum(torch.cosine_similarity(origin_embedding,
                   correspond_b_embedding, dim=1).cpu().numpy())
        avg_sim_arrack_origin_std.extend(list(torch.cosine_similarity(
            arrack_embedding, origin_embedding, dim=1).cpu().numpy()))
        avg_sim_correspond_origin_std.extend(list(torch.cosine_similarity(
            arrack_embedding, correspond_b_embedding, dim=1).cpu().numpy()))
        avg_sim_arrack_correspond_std.extend(list(torch.cosine_similarity(
            origin_embedding, correspond_b_embedding, dim=1).cpu().numpy()))
        num_count = num_count + 5

        avg_sim_arrack_origin_ncf = avg_sim_arrack_origin_ncf + \
            np.sum(torch.cosine_similarity(arrack_embedding_ncf,
                   origin_embedding_ncf, dim=1).cpu().numpy())
        avg_sim_correspond_origin_ncf = avg_sim_correspond_origin_ncf + \
            np.sum(torch.cosine_similarity(arrack_embedding_ncf,
                   correspond_b_embedding_ncf, dim=1).cpu().numpy())
        avg_sim_arrack_correspond_ncf = avg_sim_arrack_correspond_ncf + \
            np.sum(torch.cosine_similarity(origin_embedding_ncf,
                   correspond_b_embedding_ncf, dim=1).cpu().numpy())
        avg_sim_arrack_origin_std_ncf.extend(list(torch.cosine_similarity(
            arrack_embedding_ncf, origin_embedding_ncf, dim=1).cpu().numpy()))
        avg_sim_correspond_origin_std_ncf.extend(list(torch.cosine_similarity(
            arrack_embedding_ncf, correspond_b_embedding_ncf, dim=1).cpu().numpy()))
        avg_sim_arrack_correspond_std_ncf.extend(list(torch.cosine_similarity(
            origin_embedding_ncf, correspond_b_embedding_ncf, dim=1).cpu().numpy()))

        attack_rating = list(model_a_temp(
            uids, gmf_item_latent_attack, ncf_item_latent_attack).detach().cpu().numpy())
        data_label_to_test.extend(ratings_list)
        data_pred_to_test.extend(attack_rating)

        attack_rating_ncf = list(model_a_temp(
            uids, gmf_item_latent_attack, ncf_item_latent_attack).detach().cpu().numpy())
        data_label_to_test_ncf.extend(ratings_list)
        data_pred_to_test_ncf.extend(attack_rating_ncf)

    avg_sim_arrack_origin_std = np.array(avg_sim_arrack_origin_std)
    avg_sim_correspond_origin_std = np.array(avg_sim_correspond_origin_std)
    avg_sim_arrack_correspond_std = np.array(avg_sim_arrack_correspond_std)
    std1 = np.std(avg_sim_arrack_origin_std)
    std2 = np.std(avg_sim_correspond_origin_std)
    std3 = np.std(avg_sim_arrack_correspond_std)
    MAE, RMSE = test_acc(data_label_to_test, data_pred_to_test)
    avg_sim_arrack_origin = avg_sim_arrack_origin / num_count
    avg_sim_correspond_origin = avg_sim_correspond_origin / num_count
    avg_sim_arrack_correspond = avg_sim_arrack_correspond / num_count

    avg_sim_arrack_origin_std_ncf = np.array(avg_sim_arrack_origin_std_ncf)
    avg_sim_correspond_origin_std_ncf = np.array(
        avg_sim_correspond_origin_std_ncf)
    avg_sim_arrack_correspond_std_ncf = np.array(
        avg_sim_arrack_correspond_std_ncf)
    std1_ncf = np.std(avg_sim_arrack_origin_std_ncf)
    std2_ncf = np.std(avg_sim_correspond_origin_std_ncf)
    std3_ncf = np.std(avg_sim_arrack_correspond_std_ncf)
    MAE_ncf, RMSE_ncf = test_acc(data_label_to_test_ncf, data_pred_to_test_ncf)
    avg_sim_arrack_origin_ncf = avg_sim_arrack_origin_ncf / num_count
    avg_sim_correspond_origin_ncf = avg_sim_correspond_origin_ncf / num_count
    avg_sim_arrack_correspond_ncf = avg_sim_arrack_correspond_ncf / num_count

    return avg_sim_arrack_origin, avg_sim_correspond_origin, avg_sim_arrack_correspond, MAE, RMSE, std1, std2, std3, avg_sim_arrack_origin_ncf, avg_sim_correspond_origin_ncf, avg_sim_arrack_correspond_ncf, MAE_ncf, RMSE_ncf, std1_ncf, std2_ncf, std3_ncf


def attack_noise(uid_rating_dict, model_a, model_b, atrain_data_dict, scale_noise, maxn, maxm, clip_num):
    optimizer = torch.optim.Adam(model_a.parameters(), lr=0.0005)
    model_a.train()
    avg_sim_arrack_origin = 0
    avg_sim_correspond_origin = 0
    avg_sim_arrack_correspond = 0
    avg_sim_arrack_origin_std = []
    avg_sim_correspond_origin_std = []
    avg_sim_arrack_correspond_std = []
    num_count = 0
    data_label_to_test = []
    data_pred_to_test = []

    avg_sim_arrack_origin_ncf = 0
    avg_sim_correspond_origin_ncf = 0
    avg_sim_arrack_correspond_ncf = 0
    avg_sim_arrack_origin_std_ncf = []
    avg_sim_correspond_origin_std_ncf = []
    avg_sim_arrack_correspond_std_ncf = []

    data_label_to_test_ncf = []
    data_pred_to_test_ncf = []

    model_a_temp = NeuMF_attack(maxn, maxm, HIDDEN, DROP)
    for name, param in model_a.state_dict().items():

        if (name != "gmf_item_embedding.weight") and (name != "ncf_item_embedding.weight"):

            model_a_temp.state_dict()[name].copy_(model_a.state_dict()[name])

    model_a_temp.to(DEVICE)
    model_a_temp.train()

    def clip_by_l1norm(grad_temp, clip_num=clip_num):

        l1norm_temp = torch.norm(grad_temp, p=1, dim=1)

        l1norm_temp = torch.where(
            l1norm_temp > clip_num, l1norm_temp, torch.full_like(l1norm_temp, clip_num/10))
        grad_temp = ((torch.where((clip_num/l1norm_temp) > torch.ones_like(l1norm_temp), torch.ones_like(
            l1norm_temp), (clip_num/l1norm_temp))).unsqueeze(1).repeat(1, grad_temp.size()[1]))*grad_temp

        return grad_temp
    user_num = 0
    for ud in uid_rating_dict:
        print('user_num', user_num)
        user_num = user_num+1
        for vd in range(len(uid_rating_dict[ud])):

            rating = atrain_data_dict[(ud, uid_rating_dict[ud][vd])]

            true_label = rating

            vid_list = uid_rating_dict[ud]

            uid = np.expand_dims(np.array([ud], 'int32'), axis=1)
            vid = np.expand_dims(np.array([vid_list[vd]], 'int32'), axis=1)

            uid = torch.LongTensor(uid).squeeze(1).to(DEVICE)

            vid = torch.LongTensor(vid).squeeze(1).to(DEVICE)
            ratings = torch.Tensor(np.array([rating])).to(DEVICE)

            gmf_item_latent = model_a.state_dict(
            )["gmf_item_embedding.weight"][vid].to(DEVICE)
            ncf_item_latent = model_a.state_dict(
            )["ncf_item_embedding.weight"][vid].to(DEVICE)
            gmf_item_latent.requires_grad = False
            ncf_item_latent.requires_grad = False

            preds = model_a_temp(uid, gmf_item_latent, ncf_item_latent)

            mse_loss = torch.nn.MSELoss()
            loss = mse_loss(preds, ratings)

            dy_dx = torch.autograd.grad(loss, model_a_temp.parameters())
            original_dy_dx = list((_.detach().clone() for _ in dy_dx))

            loc_list = torch.zeros(size=[1, 16], dtype=torch.float)
            scale_list = torch.full([1, 16], scale_noise)
            la = torch.distributions.laplace.Laplace(
                loc=loc_list, scale=scale_list)
            la_noise = la.sample().to(DEVICE)
            original_dy_dx[0][uid] = (clip_by_l1norm(
                original_dy_dx[0][uid]/BATCH_SIZE) + la_noise)*BATCH_SIZE

            original_dy_dx[1][uid] = (clip_by_l1norm(
                original_dy_dx[1][uid]/BATCH_SIZE) + la_noise)*BATCH_SIZE

            model_attack = NeuMF_attack(maxn, maxm, HIDDEN, DROP)
            model_attack.load_state_dict(
                copy.deepcopy(model_a_temp.state_dict()))
            model_attack.to(DEVICE)
            model_attack.train()

            attack_y = Variable(torch.rand([1])*4+1)

            attack_y = attack_y.to(DEVICE)
            attack_y.requires_grad = True

            gmf_item_latent_attack = Variable(torch.rand([1, 16])).to(DEVICE)
            ncf_item_latent_attack = Variable(torch.rand([1, 16])).to(DEVICE)
            gmf_item_latent_attack.requires_grad = True
            ncf_item_latent_attack.requires_grad = True

            criterion_attack = torch.nn.MSELoss()

            optimizer_attack = torch.optim.LBFGS(
                [ncf_item_latent_attack, gmf_item_latent_attack, attack_y])

            gmf_target_attackembedding = torch.zeros([16]).to(DEVICE)
            ncf_target_attackembedding = torch.zeros([16]).to(DEVICE)
            num_epoches = 300
            for epoch in range(num_epoches):
                gmf_target_attackembedding = copy.deepcopy(
                    gmf_item_latent_attack.detach().squeeze())
                ncf_target_attackembedding = copy.deepcopy(
                    ncf_item_latent_attack.detach().squeeze())

                def closure():
                    optimizer_attack.zero_grad()

                    out = model_attack(
                        uid, gmf_item_latent_attack, ncf_item_latent_attack)

                    loss_attack = criterion_attack(out, attack_y)

                    grad_attack = torch.autograd.grad(
                        loss_attack, model_attack.parameters(), create_graph=True)

                    grad_diff = 0
                    for gx, gy in zip(grad_attack, original_dy_dx):
                        grad_diff += ((gx - gy) ** 2).sum()
                    grad_diff.backward()

                    if (epoch) % 100 == 0:
                        print(
                            "Epoch[{}/{}], loss: {:.6f}".format(epoch, num_epoches, loss_attack.data))
                        model_attack.eval()

                        arrack_embedding = gmf_item_latent_attack.detach().squeeze()
                        origin_embedding = model_a.state_dict(
                        )['gmf_item_embedding.weight'][vid].squeeze()
                        correspond_b_embedding = model_b.state_dict(
                        )['gmf_item_embedding.weight'][vid-1].squeeze()

                        print(torch.cosine_similarity(arrack_embedding,
                              origin_embedding, dim=0).cpu().numpy())
                        print(torch.cosine_similarity(
                            arrack_embedding, correspond_b_embedding, dim=0))
                        print(torch.cosine_similarity(
                            origin_embedding, correspond_b_embedding, dim=0))
                        model_attack.train()
                    return grad_diff

                optimizer_attack.step(closure)
                if ((torch.isnan(gmf_item_latent_attack.detach().squeeze()).any()) == True) or (torch.isnan(ncf_item_latent_attack.detach().squeeze()).any() == True) or ((torch.isinf(gmf_item_latent_attack.detach().squeeze()).any()) == True) or (torch.isinf(ncf_item_latent_attack.detach().squeeze()).any() == True):
                    gmf_item_latent_attack = copy.deepcopy(
                        gmf_target_attackembedding)
                    ncf_item_latent_attack = copy.deepcopy(
                        ncf_target_attackembedding)
                    gmf_item_latent_attack = gmf_item_latent_attack.unsqueeze(
                        0)
                    ncf_item_latent_attack = ncf_item_latent_attack.unsqueeze(
                        0)
                    break

            model_attack.eval()

            arrack_embedding = gmf_target_attackembedding
            origin_embedding = model_a.state_dict(
            )['gmf_item_embedding.weight'][vid].squeeze()
            correspond_b_embedding = model_b.state_dict(
            )['gmf_item_embedding.weight'][vid-1].squeeze()

            arrack_embedding_ncf = ncf_target_attackembedding
            origin_embedding_ncf = model_a.state_dict(
            )['ncf_item_embedding.weight'][vid].squeeze()
            correspond_b_embedding_ncf = model_b.state_dict(
            )['ncf_item_embedding.weight'][vid-1].squeeze()

            avg_sim_arrack_origin = avg_sim_arrack_origin + \
                torch.cosine_similarity(
                    arrack_embedding, origin_embedding, dim=0).cpu().numpy()
            avg_sim_correspond_origin = avg_sim_correspond_origin + \
                torch.cosine_similarity(
                    arrack_embedding, correspond_b_embedding, dim=0).cpu().numpy()
            avg_sim_arrack_correspond = avg_sim_arrack_correspond + \
                torch.cosine_similarity(
                    origin_embedding, correspond_b_embedding, dim=0).cpu().numpy()
            avg_sim_arrack_origin_std.append(torch.cosine_similarity(
                arrack_embedding, origin_embedding, dim=0).cpu().numpy())
            avg_sim_correspond_origin_std.append(torch.cosine_similarity(
                arrack_embedding, correspond_b_embedding, dim=0).cpu().numpy())
            avg_sim_arrack_correspond_std.append(torch.cosine_similarity(
                origin_embedding, correspond_b_embedding, dim=0).cpu().numpy())
            num_count = num_count + 1

            avg_sim_arrack_origin_ncf = avg_sim_arrack_origin_ncf + \
                torch.cosine_similarity(
                    arrack_embedding_ncf, origin_embedding_ncf, dim=0).cpu().numpy()
            avg_sim_correspond_origin_ncf = avg_sim_correspond_origin_ncf + \
                torch.cosine_similarity(
                    arrack_embedding_ncf, correspond_b_embedding_ncf, dim=0).cpu().numpy()
            avg_sim_arrack_correspond_ncf = avg_sim_arrack_correspond_ncf + \
                torch.cosine_similarity(
                    origin_embedding_ncf, correspond_b_embedding_ncf, dim=0).cpu().numpy()
            avg_sim_arrack_origin_std_ncf.append(torch.cosine_similarity(
                arrack_embedding_ncf, origin_embedding_ncf, dim=0).cpu().numpy())
            avg_sim_correspond_origin_std_ncf.append(torch.cosine_similarity(
                arrack_embedding_ncf, correspond_b_embedding_ncf, dim=0).cpu().numpy())
            avg_sim_arrack_correspond_std_ncf.append(torch.cosine_similarity(
                origin_embedding_ncf, correspond_b_embedding_ncf, dim=0).cpu().numpy())

            attack_rating = list(model_a_temp(
                uid, gmf_item_latent_attack, ncf_item_latent_attack).detach().cpu().numpy())[0]
            data_label_to_test.append(rating)
            data_pred_to_test.append(attack_rating)

            attack_rating_ncf = list(model_a_temp(
                uid, gmf_item_latent_attack, ncf_item_latent_attack).detach().cpu().numpy())[0]
            data_label_to_test_ncf.append(rating)
            data_pred_to_test_ncf.append(attack_rating_ncf)

    avg_sim_arrack_origin_std = np.array(avg_sim_arrack_origin_std)
    avg_sim_correspond_origin_std = np.array(avg_sim_correspond_origin_std)
    avg_sim_arrack_correspond_std = np.array(avg_sim_arrack_correspond_std)
    std1 = np.std(avg_sim_arrack_origin_std)
    std2 = np.std(avg_sim_correspond_origin_std)
    std3 = np.std(avg_sim_arrack_correspond_std)
    MAE, RMSE = test_acc(data_label_to_test, data_pred_to_test)
    avg_sim_arrack_origin = avg_sim_arrack_origin / num_count
    avg_sim_correspond_origin = avg_sim_correspond_origin / num_count
    avg_sim_arrack_correspond = avg_sim_arrack_correspond / num_count

    avg_sim_arrack_origin_std_ncf = np.array(avg_sim_arrack_origin_std_ncf)
    avg_sim_correspond_origin_std_ncf = np.array(
        avg_sim_correspond_origin_std_ncf)
    avg_sim_arrack_correspond_std_ncf = np.array(
        avg_sim_arrack_correspond_std_ncf)
    std1_ncf = np.std(avg_sim_arrack_origin_std_ncf)
    std2_ncf = np.std(avg_sim_correspond_origin_std_ncf)
    std3_ncf = np.std(avg_sim_arrack_correspond_std_ncf)
    MAE_ncf, RMSE_ncf = test_acc(data_label_to_test_ncf, data_pred_to_test_ncf)
    avg_sim_arrack_origin_ncf = avg_sim_arrack_origin_ncf / num_count
    avg_sim_correspond_origin_ncf = avg_sim_correspond_origin_ncf / num_count
    avg_sim_arrack_correspond_ncf = avg_sim_arrack_correspond_ncf / num_count

    return avg_sim_arrack_origin, avg_sim_correspond_origin, avg_sim_arrack_correspond, MAE, RMSE, std1, std2, std3, avg_sim_arrack_origin_ncf, avg_sim_correspond_origin_ncf, avg_sim_arrack_correspond_ncf, MAE_ncf, RMSE_ncf, std1_ncf, std2_ncf, std3_ncf


def attack_average_noise(uid_rating_dict, model_a, model_b, atrain_data_dict, scale_noise, maxn, maxm, clip_num):
    optimizer = torch.optim.Adam(model_a.parameters(), lr=0.0005)
    model_a.train()
    avg_sim_arrack_origin = 0
    avg_sim_correspond_origin = 0
    avg_sim_arrack_correspond = 0
    avg_sim_arrack_origin_std = []
    avg_sim_correspond_origin_std = []
    avg_sim_arrack_correspond_std = []
    num_count = 0
    data_label_to_test = []
    data_pred_to_test = []

    avg_sim_arrack_origin_ncf = 0
    avg_sim_correspond_origin_ncf = 0
    avg_sim_arrack_correspond_ncf = 0
    avg_sim_arrack_origin_std_ncf = []
    avg_sim_correspond_origin_std_ncf = []
    avg_sim_arrack_correspond_std_ncf = []

    data_label_to_test_ncf = []
    data_pred_to_test_ncf = []

    model_a_temp = NeuMF_attack(maxn, maxm, HIDDEN, DROP)
    for name, param in model_a.state_dict().items():

        if (name != "gmf_item_embedding.weight") and (name != "ncf_item_embedding.weight"):

            model_a_temp.state_dict()[name].copy_(model_a.state_dict()[name])

    model_a_temp.to(DEVICE)
    model_a_temp.train()

    def clip_by_l1norm(grad_temp, clip_num=clip_num):

        l1norm_temp = torch.norm(grad_temp, p=1, dim=1)

        l1norm_temp = torch.where(
            l1norm_temp > clip_num, l1norm_temp, torch.full_like(l1norm_temp, clip_num/10))
        grad_temp = ((torch.where((clip_num/l1norm_temp) > torch.ones_like(l1norm_temp), torch.ones_like(
            l1norm_temp), (clip_num/l1norm_temp))).unsqueeze(1).repeat(1, grad_temp.size()[1]))*grad_temp

        return grad_temp

    user_num = 0
    for ud in uid_rating_dict:
        print('user_num', user_num)
        user_num = user_num+1

        uid = torch.LongTensor(np.expand_dims(
            np.array([ud], 'int32'), axis=1)).to(DEVICE)
        uids = torch.LongTensor(np.expand_dims(
            np.array([ud]*5, 'int32'), axis=1)).squeeze(1).to(DEVICE)
        vids = torch.LongTensor(np.expand_dims(
            np.array(uid_rating_dict[ud], 'int32'), axis=1)).squeeze(1).to(DEVICE)
        ratings_list = []
        for vd in range(len(uid_rating_dict[ud])):
            ratings_list.append(
                atrain_data_dict[(ud, uid_rating_dict[ud][vd])])
        true_labels = ratings_list

        ratings = torch.Tensor(np.expand_dims(
            np.array(ratings_list, 'int32'), axis=1)).squeeze(1).to(DEVICE)

        gmf_item_latent = model_a.state_dict(
        )["gmf_item_embedding.weight"][vids].to(DEVICE)
        ncf_item_latent = model_a.state_dict(
        )["ncf_item_embedding.weight"][vids].to(DEVICE)
        gmf_item_latent.requires_grad = False
        ncf_item_latent.requires_grad = False

        preds = model_a_temp(uids, gmf_item_latent, ncf_item_latent)

        mse_loss = torch.nn.MSELoss()
        loss = mse_loss(preds, ratings)

        dy_dx = torch.autograd.grad(loss, model_a_temp.parameters())
        original_dy_dx = list((_.detach().clone() for _ in dy_dx))

        loc_list = torch.zeros(size=[1, 16], dtype=torch.float)
        scale_list = torch.full([1, 16], scale_noise)
        la = torch.distributions.laplace.Laplace(
            loc=loc_list, scale=scale_list)
        la_noise = la.sample().to(DEVICE)

        original_dy_dx[0][uid][0] = (clip_by_l1norm(
            original_dy_dx[0][uid][0]/BATCH_SIZE) + la_noise)*BATCH_SIZE

        original_dy_dx[1][uid][0] = (clip_by_l1norm(
            original_dy_dx[1][uid][0]/BATCH_SIZE) + la_noise)*BATCH_SIZE

        model_attack = NeuMF_attack(maxn, maxm, HIDDEN, DROP)
        model_attack.load_state_dict(copy.deepcopy(model_a_temp.state_dict()))
        model_attack.to(DEVICE)
        model_attack.train()

        attack_y = Variable(torch.rand([5])*4+1)

        attack_y = attack_y.to(DEVICE)
        attack_y.requires_grad = True

        gmf_item_latent_attack = Variable(torch.rand([5, 16])).to(DEVICE)
        ncf_item_latent_attack = Variable(torch.rand([5, 16])).to(DEVICE)
        gmf_item_latent_attack.requires_grad = True
        ncf_item_latent_attack.requires_grad = True

        criterion_attack = torch.nn.MSELoss()

        optimizer_attack = torch.optim.LBFGS(
            [ncf_item_latent_attack, gmf_item_latent_attack, attack_y])
        gmf_target_attackembedding = torch.zeros([5, 16]).to(DEVICE)
        ncf_target_attackembedding = torch.zeros([5, 16]).to(DEVICE)

        num_epoches = 300
        for epoch in range(num_epoches):
            gmf_target_attackembedding = copy.deepcopy(
                gmf_item_latent_attack.detach().squeeze())
            ncf_target_attackembedding = copy.deepcopy(
                ncf_item_latent_attack.detach().squeeze())

            def closure():

                optimizer_attack.zero_grad()

                out = model_attack(
                    uids, gmf_item_latent_attack, ncf_item_latent_attack)

                loss_attack = criterion_attack(out, attack_y)

                grad_attack = torch.autograd.grad(
                    loss_attack, model_attack.parameters(), create_graph=True)

                grad_diff = 0
                for gx, gy in zip(grad_attack, original_dy_dx):
                    grad_diff += ((gx - gy) ** 2).sum()
                grad_diff.backward()

                if (epoch) % 100 == 0:
                    print("Epoch[{}/{}], loss: {:.6f}".format(epoch,
                          num_epoches, loss_attack.data))
                    model_attack.eval()

                    arrack_embedding = gmf_item_latent_attack.detach().squeeze()
                    origin_embedding = model_a.state_dict(
                    )['gmf_item_embedding.weight'][vids].squeeze()
                    correspond_b_embedding = model_b.state_dict(
                    )['gmf_item_embedding.weight'][vids-torch.LongTensor(np.array([1]*5)).to(DEVICE)].squeeze()

                    print(torch.cosine_similarity(arrack_embedding,
                          origin_embedding, dim=1).cpu().numpy())
                    print(torch.cosine_similarity(arrack_embedding,
                          correspond_b_embedding, dim=1).cpu().numpy())
                    print(torch.cosine_similarity(origin_embedding,
                          correspond_b_embedding, dim=1).cpu().numpy())
                    model_attack.train()
                return grad_diff

            optimizer_attack.step(closure)
            if ((torch.isnan(gmf_item_latent_attack.detach().squeeze()).any()) == True) or (torch.isnan(ncf_item_latent_attack.detach().squeeze()).any() == True) or ((torch.isinf(gmf_item_latent_attack.detach().squeeze()).any()) == True) or (torch.isinf(ncf_item_latent_attack.detach().squeeze()).any() == True):
                gmf_item_latent_attack = copy.deepcopy(
                    gmf_target_attackembedding)
                ncf_item_latent_attack = copy.deepcopy(
                    ncf_target_attackembedding)

                break

        print()

        model_attack.eval()

        arrack_embedding = gmf_target_attackembedding
        origin_embedding = model_a.state_dict(
        )['gmf_item_embedding.weight'][vids].squeeze()
        correspond_b_embedding = model_b.state_dict(
        )['gmf_item_embedding.weight'][vids-torch.LongTensor(np.array([1]*5)).to(DEVICE)].squeeze()

        arrack_embedding_ncf = ncf_target_attackembedding
        origin_embedding_ncf = model_a.state_dict(
        )['ncf_item_embedding.weight'][vids].squeeze()
        correspond_b_embedding_ncf = model_b.state_dict(
        )['ncf_item_embedding.weight'][vids-torch.LongTensor(np.array([1]*5)).to(DEVICE)].squeeze()

        print(torch.cosine_similarity(arrack_embedding,
              origin_embedding, dim=1).cpu().numpy())
        print(torch.cosine_similarity(arrack_embedding,
              correspond_b_embedding, dim=1).cpu().numpy())
        print(torch.cosine_similarity(origin_embedding,
              correspond_b_embedding, dim=1).cpu().numpy())
        avg_sim_arrack_origin = avg_sim_arrack_origin + \
            np.sum(torch.cosine_similarity(arrack_embedding,
                   origin_embedding, dim=1).cpu().numpy())
        avg_sim_correspond_origin = avg_sim_correspond_origin + \
            np.sum(torch.cosine_similarity(arrack_embedding,
                   correspond_b_embedding, dim=1).cpu().numpy())
        avg_sim_arrack_correspond = avg_sim_arrack_correspond + \
            np.sum(torch.cosine_similarity(origin_embedding,
                   correspond_b_embedding, dim=1).cpu().numpy())
        avg_sim_arrack_origin_std.extend(list(torch.cosine_similarity(
            arrack_embedding, origin_embedding, dim=1).cpu().numpy()))
        avg_sim_correspond_origin_std.extend(list(torch.cosine_similarity(
            arrack_embedding, correspond_b_embedding, dim=1).cpu().numpy()))
        avg_sim_arrack_correspond_std.extend(list(torch.cosine_similarity(
            origin_embedding, correspond_b_embedding, dim=1).cpu().numpy()))
        num_count = num_count + 5

        avg_sim_arrack_origin_ncf = avg_sim_arrack_origin_ncf + \
            np.sum(torch.cosine_similarity(arrack_embedding_ncf,
                   origin_embedding_ncf, dim=1).cpu().numpy())
        avg_sim_correspond_origin_ncf = avg_sim_correspond_origin_ncf + \
            np.sum(torch.cosine_similarity(arrack_embedding_ncf,
                   correspond_b_embedding_ncf, dim=1).cpu().numpy())
        avg_sim_arrack_correspond_ncf = avg_sim_arrack_correspond_ncf + \
            np.sum(torch.cosine_similarity(origin_embedding_ncf,
                   correspond_b_embedding_ncf, dim=1).cpu().numpy())
        avg_sim_arrack_origin_std_ncf.extend(list(torch.cosine_similarity(
            arrack_embedding_ncf, origin_embedding_ncf, dim=1).cpu().numpy()))
        avg_sim_correspond_origin_std_ncf.extend(list(torch.cosine_similarity(
            arrack_embedding_ncf, correspond_b_embedding_ncf, dim=1).cpu().numpy()))
        avg_sim_arrack_correspond_std_ncf.extend(list(torch.cosine_similarity(
            origin_embedding_ncf, correspond_b_embedding_ncf, dim=1).cpu().numpy()))

        attack_rating = list(model_a_temp(
            uids, gmf_item_latent_attack, ncf_item_latent_attack).detach().cpu().numpy())
        data_label_to_test.extend(ratings_list)
        data_pred_to_test.extend(attack_rating)

        attack_rating_ncf = list(model_a_temp(
            uids, gmf_item_latent_attack, ncf_item_latent_attack).detach().cpu().numpy())
        data_label_to_test_ncf.extend(ratings_list)
        data_pred_to_test_ncf.extend(attack_rating_ncf)

    avg_sim_arrack_origin_std = np.array(avg_sim_arrack_origin_std)
    avg_sim_correspond_origin_std = np.array(avg_sim_correspond_origin_std)
    avg_sim_arrack_correspond_std = np.array(avg_sim_arrack_correspond_std)
    std1 = np.std(avg_sim_arrack_origin_std)
    std2 = np.std(avg_sim_correspond_origin_std)
    std3 = np.std(avg_sim_arrack_correspond_std)
    MAE, RMSE = test_acc(data_label_to_test, data_pred_to_test)
    avg_sim_arrack_origin = avg_sim_arrack_origin / num_count
    avg_sim_correspond_origin = avg_sim_correspond_origin / num_count
    avg_sim_arrack_correspond = avg_sim_arrack_correspond / num_count

    avg_sim_arrack_origin_std_ncf = np.array(avg_sim_arrack_origin_std_ncf)
    avg_sim_correspond_origin_std_ncf = np.array(
        avg_sim_correspond_origin_std_ncf)
    avg_sim_arrack_correspond_std_ncf = np.array(
        avg_sim_arrack_correspond_std_ncf)
    std1_ncf = np.std(avg_sim_arrack_origin_std_ncf)
    std2_ncf = np.std(avg_sim_correspond_origin_std_ncf)
    std3_ncf = np.std(avg_sim_arrack_correspond_std_ncf)
    MAE_ncf, RMSE_ncf = test_acc(data_label_to_test_ncf, data_pred_to_test_ncf)
    avg_sim_arrack_origin_ncf = avg_sim_arrack_origin_ncf / num_count
    avg_sim_correspond_origin_ncf = avg_sim_correspond_origin_ncf / num_count
    avg_sim_arrack_correspond_ncf = avg_sim_arrack_correspond_ncf / num_count

    return avg_sim_arrack_origin, avg_sim_correspond_origin, avg_sim_arrack_correspond, MAE, RMSE, std1, std2, std3, avg_sim_arrack_origin_ncf, avg_sim_correspond_origin_ncf, avg_sim_arrack_correspond_ncf, MAE_ncf, RMSE_ncf, std1_ncf, std2_ncf, std3_ncf
