

import copy
import random
import numpy as np
from loguru import logger
import torch
from torch.autograd import Variable

from const import *
from generator import *
from model_torch import MF
from preprocess import *
from attack_function import *
manual_seed = 1
random.seed(manual_seed)
np.random.seed(manual_seed)
torch.manual_seed(manual_seed)


def attack_direct(uid_rating_dict, model_a, model_b, atrain_data_dict):

    optimizer = torch.optim.SGD(model_a.parameters(), lr=1.0)
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
    for ud in uid_rating_dict:
        for vd in range(len(uid_rating_dict[ud])):

            rating = atrain_data_dict[(ud, uid_rating_dict[ud][vd])]
            vid_list = uid_rating_dict[ud]

            uid = np.expand_dims(np.array([ud], 'int32'), axis=1)
            vid = np.expand_dims(np.array([vid_list[vd]], 'int32'), axis=1)
            uid = torch.LongTensor(uid).to(DEVICE)
            vid = torch.LongTensor(vid).to(DEVICE)
            ratings = torch.Tensor(np.array([rating])).to(DEVICE).t()

            pre_model_weights_a = copy.deepcopy(model_a.state_dict())
            preds = model_a(uid, vid)

            pred_temp = torch.dot(pre_model_weights_a['uembedding_layer.weight'][uid].squeeze(
            ), pre_model_weights_a['iembedding_layer.weight'][vid].squeeze())
            err = rating - preds
            gradu = -err * \
                pre_model_weights_a['iembedding_layer.weight'][vid]*2

            mse_loss = torch.nn.MSELoss()
            loss = mse_loss(preds, ratings)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            cur_model_weights_a = copy.deepcopy(model_a.state_dict())
            model_a.load_state_dict(pre_model_weights_a)

            gradu = Variable((pre_model_weights_a['uembedding_layer.weight'][uid] -
                             cur_model_weights_a['uembedding_layer.weight'][uid]).squeeze().unsqueeze(1)).to(DEVICE)

            err_repeat = Variable(-(rating - preds.repeat(20,
                                  1).squeeze()).unsqueeze(1)).to(DEVICE)

            class LinearRegression(torch.nn.Module):
                def __init__(self):
                    super(LinearRegression, self).__init__()
                    self.arrack_iembedding = torch.nn.Parameter(
                        (torch.rand(20, 1)-0.5)*0.01)

                def forward(self, err):

                    out = self.arrack_iembedding * err * 2

                    return out

            model_attack = LinearRegression().to(DEVICE)
            criterion_attack = torch.nn.MSELoss()
            optimizer_attack = torch.optim.Adam(
                model_attack.parameters(), lr=1e-2)
            model_attack.train()
            err_repeat = err_repeat.to(DEVICE)
            gradu = gradu.to(DEVICE)

            num_epoches = 1000
            for epoch in range(num_epoches):

                out = model_attack(err_repeat)

                loss_attack = criterion_attack(out, gradu)

                optimizer_attack.zero_grad()
                loss_attack.backward(retain_graph=True)
                optimizer_attack.step()

                if (epoch) % 999 == 0:
                    print("Epoch[{}/{}], loss: {:.6f}".format(epoch,
                          num_epoches, loss_attack.data))

            model_attack.eval()

            arrack_embedding = model_attack.state_dict()[
                'arrack_iembedding'].squeeze()
            origin_embedding = model_a.state_dict(
            )['iembedding_layer.weight'][vid].squeeze()
            correspond_b_embedding = model_b.state_dict(
            )['iembedding_layer.weight'][vid-1].squeeze()

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

            attack_rating = torch.dot(pre_model_weights_a['uembedding_layer.weight'][uid].squeeze(
            ), arrack_embedding).cpu().numpy()

            data_label_to_test.append(rating)
            data_pred_to_test.append(attack_rating)

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

    return avg_sim_arrack_origin, avg_sim_correspond_origin, avg_sim_arrack_correspond, MAE, RMSE, std1, std2, std3


def attack_average(uid_rating_dict, model_a, model_b, atrain_data_dict):

    optimizer = torch.optim.SGD(model_a.parameters(), lr=1.0)
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

    temp_num = 50
    for ud in uid_rating_dict: 

        uid = torch.LongTensor(np.expand_dims(
            np.array([ud], 'int32'), axis=1)).to(DEVICE)
        uids = torch.LongTensor(np.expand_dims(
            np.array([ud]*5, 'int32'), axis=1)).to(DEVICE)
        vids = torch.LongTensor(np.expand_dims(
            np.array(uid_rating_dict[ud], 'int32'), axis=1)).to(DEVICE)
        ratings_list = []
        for vd in range(len(uid_rating_dict[ud])):
            ratings_list.append(
                atrain_data_dict[(ud, uid_rating_dict[ud][vd])])
        ratings = torch.Tensor(np.expand_dims(
            np.array(ratings_list, 'int32'), axis=1)).to(DEVICE)

        pre_model_weights_a = copy.deepcopy(model_a.state_dict())

        preds = model_a(uids, vids)

        mse_loss = torch.nn.MSELoss()
        loss = mse_loss(preds, ratings)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        cur_model_weights_a = copy.deepcopy(model_a.state_dict())
        model_a.load_state_dict(pre_model_weights_a)

        gradu = Variable((pre_model_weights_a['uembedding_layer.weight'][uid] -
                         cur_model_weights_a['uembedding_layer.weight'][uid]).squeeze().unsqueeze(1)).to(DEVICE)

        err_repeat_list = []
        for i in range(5):
            err_repeat = Variable(-(ratings[i] - preds[i].repeat(
                20, 1).squeeze()).unsqueeze(1)).to(DEVICE)
            err_repeat_list.append(err_repeat)

        class LinearRegression2(torch.nn.Module):
            def __init__(self):
                super(LinearRegression2, self).__init__()

                self.arrack_iembedding0 = torch.nn.Parameter(
                    (torch.rand(20, 1)-0.5)*0.01)
                self.arrack_iembedding1 = torch.nn.Parameter(
                    (torch.rand(20, 1)-0.5)*0.01)
                self.arrack_iembedding2 = torch.nn.Parameter(
                    (torch.rand(20, 1)-0.5)*0.01)
                self.arrack_iembedding3 = torch.nn.Parameter(
                    (torch.rand(20, 1)-0.5)*0.01)
                self.arrack_iembedding4 = torch.nn.Parameter(
                    (torch.rand(20, 1)-0.5)*0.01)

            def forward(self, err_list):

                out = (self.arrack_iembedding0 * err_list[0]*2+self.arrack_iembedding1 * err_list[1]*2+self.arrack_iembedding2 *
                       err_list[2]*2+self.arrack_iembedding3 * err_list[3]*2+self.arrack_iembedding4 * err_list[4]*2)/5

                return out

        model_attack = LinearRegression2().to(DEVICE)
        criterion_attack = torch.nn.MSELoss()
        optimizer_attack = torch.optim.Adam(model_attack.parameters(), lr=1e-2)
        model_attack.train()

        num_epoches = 1000
        for epoch in range(num_epoches):

            out = model_attack(err_repeat_list)

            loss_attack = criterion_attack(out, gradu)

            optimizer_attack.zero_grad()
            loss_attack.backward(retain_graph=True)
            optimizer_attack.step()

            if (epoch) % 999 == 0:
                print("Epoch[{}/{}], loss: {:.6f}".format(epoch,
                      num_epoches, loss_attack.data))

        model_attack.eval()

        for i in range(5):

            arrack_embedding = model_attack.state_dict(
            )['arrack_iembedding'+str(i)].squeeze()
            origin_embedding = model_a.state_dict(
            )['iembedding_layer.weight'][vids[i]].squeeze()
            correspond_b_embedding = model_b.state_dict(
            )['iembedding_layer.weight'][vids[i]-1].squeeze()

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

            attack_rating = torch.dot(pre_model_weights_a['uembedding_layer.weight'][uid].squeeze(
            ), arrack_embedding).cpu().numpy()

            data_label_to_test.append(ratings_list[i])
            data_pred_to_test.append(attack_rating)

        temp_num = temp_num-1
        if temp_num == 0:
            break

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

    return avg_sim_arrack_origin, avg_sim_correspond_origin, avg_sim_arrack_correspond, MAE, RMSE, std1, std2, std3


def attack_noise(uid_rating_dict, model_a, model_b, atrain_data_dict, scale_noise, clip_num):

    optimizer = torch.optim.SGD(model_a.parameters(), lr=1.0)
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
    for ud in uid_rating_dict:
        for vd in range(len(uid_rating_dict[ud])):

            rating = atrain_data_dict[(ud, uid_rating_dict[ud][vd])]
            vid_list = uid_rating_dict[ud]

            uid = np.expand_dims(np.array([ud], 'int32'), axis=1)
            vid = np.expand_dims(np.array([vid_list[vd]], 'int32'), axis=1)
            uid = torch.LongTensor(uid).to(DEVICE)
            vid = torch.LongTensor(vid).to(DEVICE)
            ratings = torch.Tensor(np.array([rating])).to(DEVICE).t()

            pre_model_weights_a = copy.deepcopy(model_a.state_dict())
            preds = model_a(uid, vid)

            pred_temp = torch.dot(pre_model_weights_a['uembedding_layer.weight'][uid].squeeze(
            ), pre_model_weights_a['iembedding_layer.weight'][vid].squeeze())
            err = rating - preds
            gradu = -err * \
                pre_model_weights_a['iembedding_layer.weight'][vid]*2

            mse_loss = torch.nn.MSELoss()
            loss = mse_loss(preds, ratings)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            cur_model_weights_a = copy.deepcopy(model_a.state_dict())
            model_a.load_state_dict(pre_model_weights_a)

            loc_list = torch.zeros(size=[20, 1], dtype=torch.float)
            scale_list = torch.full([20, 1], scale_noise)
            la = torch.distributions.laplace.Laplace(
                loc=loc_list, scale=scale_list)

            def clip_by_l1norm(grad_temp, clip_num=clip_num):

                l1norm_temp = torch.norm(grad_temp, p=1, dim=0)

                l1norm_temp = torch.where(
                    l1norm_temp > clip_num, l1norm_temp, torch.full_like(l1norm_temp, clip_num/10))
                grad_temp = ((torch.where((clip_num/l1norm_temp) > torch.ones_like(l1norm_temp), torch.ones_like(
                    l1norm_temp), (clip_num/l1norm_temp))).unsqueeze(1).repeat(grad_temp.size()[0], 1)) * grad_temp

                return grad_temp

            gradutemp = ((pre_model_weights_a['uembedding_layer.weight'][uid] -
                         cur_model_weights_a['uembedding_layer.weight'][uid]).squeeze().unsqueeze(1))

            gradu = Variable(clip_by_l1norm(gradutemp/BATCH_SIZE) *
                             BATCH_SIZE + (la.sample()*BATCH_SIZE).to(DEVICE)).to(DEVICE)

            err_repeat = Variable(-(rating - preds.repeat(20,
                                  1).squeeze()).unsqueeze(1)).to(DEVICE)

            class LinearRegression(torch.nn.Module):
                def __init__(self):
                    super(LinearRegression, self).__init__()

                    self.arrack_iembedding = torch.nn.Parameter(
                        (torch.rand(20, 1)-0.5)*0.01)

                def forward(self, err):
                    out = self.arrack_iembedding * err * 2

                    return out

            model_attack = LinearRegression().to(DEVICE)
            criterion_attack = torch.nn.MSELoss()
            optimizer_attack = torch.optim.Adam(
                model_attack.parameters(), lr=1e-2)
            model_attack.train()
            err_repeat = err_repeat.to(DEVICE)
            gradu = gradu.to(DEVICE)

            num_epoches = 1000
            for epoch in range(num_epoches):

                out = model_attack(err_repeat)

                loss_attack = criterion_attack(out, gradu)

                optimizer_attack.zero_grad()
                loss_attack.backward(retain_graph=True)
                optimizer_attack.step()

                if (epoch) % 999 == 0:
                    print("Epoch[{}/{}], loss: {:.6f}".format(epoch,
                          num_epoches, loss_attack.data))

            model_attack.eval()

            arrack_embedding = model_attack.state_dict()[
                'arrack_iembedding'].squeeze()
            origin_embedding = model_a.state_dict(
            )['iembedding_layer.weight'][vid].squeeze()
            correspond_b_embedding = model_b.state_dict(
            )['iembedding_layer.weight'][vid-1].squeeze()

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

            attack_rating = torch.dot(pre_model_weights_a['uembedding_layer.weight'][uid].squeeze(
            ), arrack_embedding).cpu().numpy()

            data_label_to_test.append(rating)
            data_pred_to_test.append(attack_rating)

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

    return avg_sim_arrack_origin, avg_sim_correspond_origin, avg_sim_arrack_correspond, MAE, RMSE, std1, std2, std3


def attack_average_noise(uid_rating_dict, model_a, model_b, atrain_data_dict, scale_noise, clip_num):

    optimizer = torch.optim.SGD(model_a.parameters(), lr=1.0)
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

    temp_num = 50
    for ud in uid_rating_dict:

        uid = torch.LongTensor(np.expand_dims(
            np.array([ud], 'int32'), axis=1)).to(DEVICE)
        uids = torch.LongTensor(np.expand_dims(
            np.array([ud]*5, 'int32'), axis=1)).to(DEVICE)
        vids = torch.LongTensor(np.expand_dims(
            np.array(uid_rating_dict[ud], 'int32'), axis=1)).to(DEVICE)
        ratings_list = []
        for vd in range(len(uid_rating_dict[ud])):
            ratings_list.append(
                atrain_data_dict[(ud, uid_rating_dict[ud][vd])])
        ratings = torch.Tensor(np.expand_dims(
            np.array(ratings_list, 'int32'), axis=1)).to(DEVICE)

        pre_model_weights_a = copy.deepcopy(model_a.state_dict())

        preds = model_a(uids, vids)
        mse_loss = torch.nn.MSELoss()
        loss = mse_loss(preds, ratings)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        cur_model_weights_a = copy.deepcopy(model_a.state_dict())
        model_a.load_state_dict(pre_model_weights_a)

        loc_list = torch.zeros(size=[20, 1], dtype=torch.float)
        scale_list = torch.full([20, 1], scale_noise)
        la = torch.distributions.laplace.Laplace(
            loc=loc_list, scale=scale_list)

        def clip_by_l1norm(grad_temp, clip_num=clip_num):

            l1norm_temp = torch.norm(grad_temp, p=1, dim=0)
            l1norm_temp = torch.where(
                l1norm_temp > clip_num, l1norm_temp, torch.full_like(l1norm_temp, clip_num/10))
            grad_temp = ((torch.where((clip_num/l1norm_temp) > torch.ones_like(l1norm_temp), torch.ones_like(
                l1norm_temp), (clip_num/l1norm_temp))).unsqueeze(1).repeat(grad_temp.size()[0], 1)) * grad_temp

            return grad_temp
        gradutemp = ((pre_model_weights_a['uembedding_layer.weight'][uid] -
                     cur_model_weights_a['uembedding_layer.weight'][uid]).squeeze().unsqueeze(1))

        gradu = Variable(clip_by_l1norm(gradutemp/BATCH_SIZE) *
                         BATCH_SIZE + (la.sample()*BATCH_SIZE).to(DEVICE)).to(DEVICE)

        err_repeat_list = []
        for i in range(5):
            err_repeat = Variable(-(ratings[i] - preds[i].repeat(
                20, 1).squeeze()).unsqueeze(1)).to(DEVICE)
            err_repeat_list.append(err_repeat)

        class LinearRegression2(torch.nn.Module):
            def __init__(self):
                super(LinearRegression2, self).__init__()
                self.arrack_iembedding0 = torch.nn.Parameter(
                    (torch.rand(20, 1)-0.5)*0.01)
                self.arrack_iembedding1 = torch.nn.Parameter(
                    (torch.rand(20, 1)-0.5)*0.01)
                self.arrack_iembedding2 = torch.nn.Parameter(
                    (torch.rand(20, 1)-0.5)*0.01)
                self.arrack_iembedding3 = torch.nn.Parameter(
                    (torch.rand(20, 1)-0.5)*0.01)
                self.arrack_iembedding4 = torch.nn.Parameter(
                    (torch.rand(20, 1)-0.5)*0.01)

            def forward(self, err_list):

                out = (self.arrack_iembedding0 * err_list[0]*2+self.arrack_iembedding1 * err_list[1]*2+self.arrack_iembedding2 *
                       err_list[2]*2+self.arrack_iembedding3 * err_list[3]*2+self.arrack_iembedding4 * err_list[4]*2)/5

                return out

        model_attack = LinearRegression2().to(DEVICE)
        criterion_attack = torch.nn.MSELoss()
        optimizer_attack = torch.optim.Adam(model_attack.parameters(), lr=1e-2)
        model_attack.train()

        num_epoches = 1000
        for epoch in range(num_epoches):

            out = model_attack(err_repeat_list)

            loss_attack = criterion_attack(out, gradu)

            optimizer_attack.zero_grad()
            loss_attack.backward(retain_graph=True)
            optimizer_attack.step()

            if (epoch) % 999 == 0:
                print("Epoch[{}/{}], loss: {:.6f}".format(epoch,
                      num_epoches, loss_attack.data))

        model_attack.eval()

        for i in range(5):

            arrack_embedding = model_attack.state_dict(
            )['arrack_iembedding'+str(i)].squeeze()
            origin_embedding = model_a.state_dict(
            )['iembedding_layer.weight'][vids[i]].squeeze()
            correspond_b_embedding = model_b.state_dict(
            )['iembedding_layer.weight'][vids[i]-1].squeeze()

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
            attack_rating = torch.dot(pre_model_weights_a['uembedding_layer.weight'][uid].squeeze(
            ), arrack_embedding).cpu().numpy()
            data_label_to_test.append(ratings_list[i])
            data_pred_to_test.append(attack_rating)

        temp_num = temp_num-1
        if temp_num == 0:
            break

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

    return avg_sim_arrack_origin, avg_sim_correspond_origin, avg_sim_arrack_correspond, MAE, RMSE, std1, std2, std3
