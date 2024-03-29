from attack_function import *
from preprocess import *
from model_torch import MF
from generator import *
from const import *
import sklearn
from torch.autograd import Variable
import itertools as it
import torch
from tqdm import tqdm
import numpy as np
import math
import time
import random
from pickle import GLOBAL
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
parser.add_argument("--copy",
                    type=str,
                    default='1',
                    help="copy idnex")


parser.add_argument("--lr",
                    type=float,
                    default=0.40,
                    help="learning rate")
parser.add_argument("--gpu",
                    type=str,
                    default="1",
                    help="gpu card ID")
parser.add_argument("--attacktpye",
                    type=str,
                    default="1",
                    help="attacktpye")
parser.add_argument("--type",
                    type=str,
                    default="valid",
                    help="testtype")
parser.add_argument("--eps",
                    type=str,
                    default="10",
                    help="eps")
parser.add_argument("--clipnum",
                    type=str,
                    default="0.006",
                    help="clip_num")


args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


manual_seed = 1
random.seed(manual_seed)
np.random.seed(manual_seed)
torch.manual_seed(manual_seed)

LR = args.lr
dataset = args.dataset
copy = args.copy
testtype = args.type
attacktpye = args.attacktpye

timestamp = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))

if args.eps == '0':
    scale_temp = " "
else:
    scale_temp = float(args.clipnum)/float(args.eps)
clip_num = float(args.clipnum)

dataset_temp = ''
if dataset[-4:] == 'OCCF':
    dataset_temp = dataset[:-4]
else:
    dataset_temp = dataset[:-2]

if (dataset_temp == 'ML1M') or (dataset_temp == 'ML100K'):
    atrain_data_filename = './datasets/'+dataset + \
        '/'+dataset_temp+'_copy'+copy + '_train-a'
    atest_data_filename_uni = './datasets/'+dataset+'/' + \
        dataset_temp+'_copy'+copy + '_' + testtype + '-a-uni'
    atest_data_filename_com = './datasets/'+dataset+'/' + \
        dataset_temp+'_copy'+copy + '_' + testtype + '-a-com'
    atest_data_filename_non = './datasets/'+dataset+'/' + \
        dataset_temp+'_copy'+copy + '_' + testtype + '-a-non'
    btrain_data_filename = './datasets/'+dataset + \
        '/'+dataset_temp+'_copy'+copy + '_train-b'
    btest_data_filename_uni = './datasets/'+dataset+'/' + \
        dataset_temp+'_copy'+copy + '_' + testtype + '-b-uni'
    btest_data_filename_com = './datasets/'+dataset+'/' + \
        dataset_temp+'_copy'+copy + '_' + testtype + '-b-com'
    btest_data_filename_non = './datasets/'+dataset+'/' + \
        dataset_temp+'_copy'+copy + '_' + testtype + '-b-non'
else:
    atrain_data_filename = './datasets/'+dataset+'/copy'+copy + '_train-a'
    atest_data_filename_uni = './datasets/'+dataset + \
        '/copy'+copy + '_' + testtype + '-a-uni'
    atest_data_filename_com = './datasets/'+dataset + \
        '/copy'+copy + '_' + testtype + '-a-com'
    atest_data_filename_non = './datasets/'+dataset + \
        '/copy'+copy + '_' + testtype + '-a-non'
    btrain_data_filename = './datasets/'+dataset+'/copy'+copy + '_train-b'
    btest_data_filename_uni = './datasets/'+dataset + \
        '/copy'+copy + '_' + testtype + '-b-uni'
    btest_data_filename_com = './datasets/'+dataset + \
        '/copy'+copy + '_' + testtype + '-b-com'
    btest_data_filename_non = './datasets/'+dataset + \
        '/copy'+copy + '_' + testtype + '-b-non'


exp_dir = f"exp/{dataset}-{timestamp}-copy{copy}-lr{LR}"
os.makedirs(exp_dir, exist_ok=True)
log_filename = f"{exp_dir}/log.txt"
log_file = logger.add(log_filename)
logger.info("FedCORE")
logger.info(f"Dataset: {dataset}")
logger.info(f"Experiment directory: {exp_dir}")
logger.info("Configuration")
with open("const.py", "r", encoding='utf-8') as f:
    for line in f:
        logger.info(line.strip())


SaveUV_dir = f"exp2-2/{dataset}-{timestamp}-copy{copy}-lr{LR}"


def train(model, optim, scheduler, input_data, test_data_uni, test_data_com, test_data_non):
    """
    The function conducts training process.
    """
    train_user_index, trainu, traini, trainlabel = input_data

    mse_loss = torch.nn.MSELoss()
    model.train()

    for ep in range(EPOCH):

        logger.info(f"Begin epoch {ep}...")

        traingen = generate_batch_data_random(
            BATCH_SIZE, train_user_index, trainu, traini, trainlabel)
        cnt = 0
        batchloss = []
        for i in tqdm((traingen)):
            time1 = time.time()
            userid_input, itemid_input = i[0]
            userid_input = torch.LongTensor(userid_input).to(DEVICE)
            itemid_input = torch.LongTensor(itemid_input).to(DEVICE)
            ratings = torch.Tensor(i[1]).to(DEVICE).t()

            logger.debug(
                f"converting into the GPU costs {time.time() - time1}")
            time2 = time.time()
            preds = model(userid_input, itemid_input)

            logger.debug(f"forward costs {time.time() - time2}")
            loss = mse_loss(preds, ratings)
            optim.zero_grad()
            loss.backward()

            optim.step()
            logger.debug(f"forward and update costs {time.time() - time2}")
            batchloss.append(loss.data.item())

            cnt += 1
            if cnt % PRINT_INTERVAL == 0:
                batchloss = np.array(batchloss)
                temploss = np.sum(batchloss)
                logger.info(f"Num iters {cnt} loss: {temploss:.4f}")
                batchloss = []
                print(
                    'ep = '+str(ep)+'aaa=============================================================aaa')
                test(model, test_data_uni)
                test(model, test_data_com)
                test(model, test_data_non)
            if cnt == len(trainu)//BATCH_SIZE:
                break
        global LR
        LR = LR * 0.9

        logger.info(f"Finish epoch {ep}, and begin to test...")
        print('ep = '+str(ep) +
              'aaa=============================================================aaa')
        test(model, test_data_uni)
        test(model, test_data_com)
        MAE, RMSE = test(model, test_data_non)

    return MAE, RMSE


def indexlistto01vector(indexlist, large_num):
    resul = np.zeros(large_num+3, dtype='int32')
    resul[indexlist] = 1
    return resul


def train_together(model_a, optim_a, scheduler_a, model_b, optim_b, scheduler_b, atrain_data, atest_data_uni, atest_data_com, atest_data_non, btrain_data, btest_data_uni, btest_data_com, btest_data_non, id_vector, maxn, maxm, id_set, id_change):

    train_user_index_a, trainu_a, traini_a, trainlabel_a = atrain_data

    train_user_index_b, trainu_b, traini_b, trainlabel_b = btrain_data

    [data1useruni_set, data2useruni_set, datausercom_set,
        data1itemuni_set, data2itemuni_set, dataitemcom_set] = id_set

    [iidfind, data1itemchange, data2itemchange] = id_change
    tset = set()
    tset.add((0, 0))
    tset.add((0, 1))
    tset.add((0, 5))
    tset.add((0, 20))
    tset.add((1, 0))
    tset.add((5, 0))
    tset.add((20, 0))
    tset.add((49, 0))

    data1useruni, data2useruni, datausercom, data1itemuni, data2itemuni, dataitemcom = id_vector
    data1useruni = torch.LongTensor(data1useruni).to(DEVICE)
    data2useruni = torch.LongTensor(data2useruni).to(DEVICE)
    datausercom = torch.LongTensor(datausercom).to(DEVICE)
    data1itemuni = torch.LongTensor(data1itemuni).to(DEVICE)
    data2itemuni = torch.LongTensor(data2itemuni).to(DEVICE)
    dataitemcom = torch.LongTensor(dataitemcom).to(DEVICE)

    mse_loss_a = torch.nn.MSELoss()
    model_a.train()

    mse_loss_b = torch.nn.MSELoss()
    model_b.train()

    for ep in range(EPOCH):

        logger.info(f"Begin epoch {ep}...")

        iter_num = max(len(trainu_a)//BATCH_SIZE, len(trainu_b)//BATCH_SIZE)
        traingen = generate_batch_data_randomCo(
            BATCH_SIZE, atrain_data, btrain_data, iter_num)

        cnt = 0
        batchloss_a = []
        batchloss_b = []
        small_ep = 0
        for i in tqdm((traingen)):
            import copy
            if (ep, small_ep) in tset:
                torch.save(copy.deepcopy(model_a.state_dict()),
                           SaveUV_dir + f"-partner{'a'}-EP{ep}-ep{small_ep}")
                torch.save(copy.deepcopy(model_b.state_dict()),
                           SaveUV_dir + f"-partner{'b'}-EP{ep}-ep{small_ep}")

            time1 = time.time()

            userid_input_a, itemid_input_a = i[0]
            userid_a_vector = indexlistto01vector(userid_input_a, maxn)
            itemid_a_vector = indexlistto01vector(itemid_input_a, maxm)
            userid_a_vector = torch.LongTensor(userid_a_vector).to(DEVICE)
            itemid_a_vector = torch.LongTensor(itemid_a_vector).to(DEVICE)

            userid_input_a = torch.LongTensor(userid_input_a).to(DEVICE)
            itemid_input_a = torch.LongTensor(itemid_input_a).to(DEVICE)
            ratings_a = torch.Tensor(i[1]).to(DEVICE).t()

            preds_a = model_a(userid_input_a, itemid_input_a)

            userid_input_b, itemid_input_b = i[2]
            userid_b_vector = indexlistto01vector(userid_input_b, maxn)
            itemid_b_vector = indexlistto01vector(itemid_input_b, maxm)
            userid_b_vector = torch.LongTensor(userid_b_vector).to(DEVICE)
            itemid_b_vector = torch.LongTensor(itemid_b_vector).to(DEVICE)

            userid_input_b = torch.LongTensor(userid_input_b).to(DEVICE)
            itemid_input_b = torch.LongTensor(itemid_input_b).to(DEVICE)
            ratings_b = torch.Tensor(i[3]).to(DEVICE).t()

            preds_b = model_b(userid_input_b, itemid_input_b)

            time2 = time.time()

            logger.debug(f"forward costs {time.time() - time2}")

            pre_model_weights_a = copy.deepcopy(model_a.state_dict())
            loss_a = mse_loss_a(preds_a, ratings_a)
            optim_a.zero_grad()
            loss_a.backward()

            optim_a.step()
            logger.debug(f"forward and update costs {time.time() - time2}")
            batchloss_a.append(loss_a.data.item())
            cur_model_weights_a = copy.deepcopy(model_a.state_dict())

            pre_model_weights_b = copy.deepcopy(model_b.state_dict())
            loss_b = mse_loss_b(preds_b, ratings_b)
            optim_b.zero_grad()
            loss_b.backward()

            optim_b.step()
            logger.debug(f"forward and update costs {time.time() - time2}")
            batchloss_b.append(loss_b.data.item())
            cur_model_weights_b = copy.deepcopy(model_b.state_dict())

            for j in cur_model_weights_a.keys():

                model_a_grads = pre_model_weights_a[j]-cur_model_weights_a[j]
                model_b_grads = pre_model_weights_b[j]-cur_model_weights_b[j]
                if j == 'uembedding_layer.weight':

                    logger.debug(f"data1useruni size: {data1useruni.size()}")
                    logger.debug(
                        f"userid_a_vector size: {userid_a_vector.size()}")
                    logger.debug(
                        f"model_a_grads input size: {model_a_grads.size()}")
                    logger.debug(
                        f"data1useruni input sum: {data1useruni.sum()}")

                    logger.debug(
                        f"userid_a_vector input sum: {userid_a_vector.sum()}")

                    logger.debug(
                        f"data1useruni * userid_a_vector size: {(data1useruni * userid_a_vector).size()}")
                    logger.debug(
                        f"data1useruni * userid_a_vector input sum: {(data1useruni * userid_a_vector).sum()}")

                    logger.debug(
                        f"data1usercom * userid_a_vector size: {(datausercom * userid_a_vector).size()}")
                    logger.debug(
                        f"data1usercom * userid_a_vector input sum: {(datausercom * userid_a_vector).sum()}")

                    logger.debug(
                        f"data2useruni * userid_a_vector size: {(data2useruni * userid_a_vector).size()}")
                    logger.debug(
                        f"data2useruni * userid_a_vector input sum: {(data2useruni * userid_a_vector).sum()}")

                    pre_model_weights_a[j] = pre_model_weights_a[j] - (
                        (data1useruni * userid_a_vector).unsqueeze(1).expand(-1, HIDDEN))*model_a_grads
                    pre_model_weights_b[j] = pre_model_weights_b[j] - (
                        (data2useruni * userid_b_vector).unsqueeze(1).expand(-1, HIDDEN))*model_b_grads

                    avg_grads = ((((userid_a_vector*userid_b_vector).unsqueeze(1).expand(-1, HIDDEN))*model_a_grads +
                                 ((userid_a_vector*userid_b_vector).unsqueeze(1).expand(-1, HIDDEN))*model_b_grads)/2)
                    logger.debug(
                        f"userid_a_vector*userid_b_vector size: {(userid_a_vector*userid_b_vector).size()}")
                    logger.debug(
                        f"userid_a_vector*userid_b_vector input sum: {(userid_a_vector*userid_b_vector).sum()}")

                    pre_model_weights_a[j] = pre_model_weights_a[j] - avg_grads
                    pre_model_weights_b[j] = pre_model_weights_b[j] - avg_grads

                    a_temp = datausercom * userid_a_vector
                    logger.debug(f"a_temp size: {(a_temp).size()}")
                    logger.debug(f"a_temp input sum: {(a_temp).sum()}")

                    a_temp[torch.where(
                        (userid_a_vector*userid_b_vector) == 1)] = 0
                    logger.debug(f"a_temp size: {(a_temp).size()}")
                    logger.debug(f"a_temp input sum: {(a_temp).sum()}")

                    a_temp = a_temp.unsqueeze(1).expand(-1, HIDDEN)
                    pre_model_weights_a[j] = pre_model_weights_a[j] - \
                        a_temp*model_a_grads
                    pre_model_weights_b[j] = pre_model_weights_b[j] - \
                        a_temp*model_a_grads

                    b_temp = datausercom * userid_b_vector
                    b_temp[torch.where(
                        (userid_a_vector*userid_b_vector) == 1)] = 0
                    b_temp = b_temp.unsqueeze(1).expand(-1, HIDDEN)
                    pre_model_weights_a[j] = pre_model_weights_a[j] - \
                        b_temp*model_b_grads
                    pre_model_weights_b[j] = pre_model_weights_b[j] - \
                        b_temp*model_b_grads

                elif j == 'iembedding_layer.weight':

                    pre_model_weights_a[j] = pre_model_weights_a[j] - (
                        (data1itemuni * itemid_a_vector).unsqueeze(1).expand(-1, HIDDEN))*model_a_grads
                    pre_model_weights_b[j] = pre_model_weights_b[j] - (
                        (data2itemuni * itemid_b_vector).unsqueeze(1).expand(-1, HIDDEN))*model_b_grads

                    avg_grads = ((((itemid_a_vector*itemid_b_vector).unsqueeze(1).expand(-1, HIDDEN))*model_a_grads +
                                 ((itemid_a_vector*itemid_b_vector).unsqueeze(1).expand(-1, HIDDEN))*model_b_grads)/2)
                    pre_model_weights_a[j] = pre_model_weights_a[j] - avg_grads
                    pre_model_weights_b[j] = pre_model_weights_b[j] - avg_grads

                    a_temp = dataitemcom * itemid_a_vector
                    a_temp[torch.where(
                        (itemid_a_vector*itemid_b_vector) == 1)] = 0
                    a_temp = a_temp.unsqueeze(1).expand(-1, HIDDEN)
                    pre_model_weights_a[j] = pre_model_weights_a[j] - \
                        a_temp*model_a_grads
                    pre_model_weights_b[j] = pre_model_weights_b[j] - \
                        a_temp*model_a_grads
                    b_temp = dataitemcom * itemid_b_vector
                    b_temp[torch.where(
                        (itemid_a_vector*itemid_b_vector) == 1)] = 0
                    b_temp = b_temp.unsqueeze(1).expand(-1, HIDDEN)
                    pre_model_weights_a[j] = pre_model_weights_a[j] - \
                        b_temp*model_b_grads
                    pre_model_weights_b[j] = pre_model_weights_b[j] - \
                        b_temp*model_b_grads

                else:

                    pre_model_weights_a[j] = pre_model_weights_a[j] - \
                        (model_a_grads + model_b_grads)/2
                    pre_model_weights_b[j] = pre_model_weights_b[j] - \
                        (model_a_grads + model_b_grads)/2

            small_ep = small_ep + 1
            model_a.load_state_dict(pre_model_weights_a)
            model_b.load_state_dict(pre_model_weights_b)

            cnt += 1
            if cnt % PRINT_INTERVAL == 0:
                batchloss_a = np.array(batchloss_a)
                temploss_a = np.sum(batchloss_a)
                logger.info(
                    f"Num iters partner a {cnt} loss: {temploss_a:.4f}")
                batchloss_a = []
                print(
                    'ep = '+str(ep)+'aaa=============================================================aaa')
                test(model_a, atest_data_uni)
                test(model_a, atest_data_com)
                test(model_a, atest_data_non)
                batchloss_b = np.array(batchloss_b)
                temploss_b = np.sum(batchloss_b)
                logger.info(
                    f"Num iters partner b {cnt} loss: {temploss_b:.4f}")
                batchloss_b = []
                print(
                    'ep = '+str(ep)+'bbb=============================================================bbb')
                test(model_b, btest_data_uni)
                test(model_b, btest_data_com)
                test(model_b, btest_data_non)
            if cnt == iter_num-1:
                break
        global LR
        LR = LR * 0.9

        logger.info(f"Finish epoch {ep}, and begin to test...")
        print('ep = '+str(ep) +
              '=============================================================')
        test(model_a, atest_data_uni)
        test(model_a, atest_data_com)
        test(model_a, atest_data_non)
        test(model_b, btest_data_uni)
        test(model_b, btest_data_com)
        test(model_b, btest_data_non)
        print()
    return 0


def test(model, input_data):
    """Test the model"""
    model.eval()
    testu, testi, testlabel = input_data
    testu = np.expand_dims(testu, axis=1)
    testi = np.expand_dims(testi, axis=1)

    userid_input = torch.LongTensor(testu).to(DEVICE)
    itemid_input = torch.LongTensor(testi).to(DEVICE)
    ratings = torch.Tensor(testlabel).to(DEVICE).t()

    preds = model(userid_input, itemid_input).squeeze()

    preds = torch.where(preds > 5, torch.full_like(preds, 5), preds)
    preds = torch.where(preds < 1, torch.full_like(preds, 1), preds)

    loss = torch.sqrt(torch.mean(torch.square(
        preds - ratings/LABEL_SCALE)))*LABEL_SCALE

    total_loss = loss.data.item()
    logger.info(f"The RMSE score is {total_loss:.4f}")
    model.train()

    ratings = ratings.to('cpu').detach().numpy()
    preds = preds.to('cpu').detach().numpy()

    max_rating = 5
    min_rating = 1
    mae = 0.0
    rmse = 0.0
    num = 0
    for i in range(len(testu)):

        pred = preds[i]

        err = pred - ratings[i]

        if np.isnan(err):
            num = num + 1
        else:
            mae = mae + abs(err)
            rmse = rmse + err*err

    print("NAN num = ", num)
    MAE = mae / len(testu)
    RMSE = math.sqrt(rmse/len(testu))
    print("RMSE: ", np.around(RMSE, 4), ' | MAE: ', np.around(MAE, 4))
    return MAE, RMSE


def item_id_tran(maxn, maxm, data1useruni_list, data2useruni_list, datausercom_list, data1itemuni_list, data2itemuni_list, dataitemcom_list, atrain_data, atest_data_uni, atest_data_com, atest_data_non, btrain_data, btest_data_uni, btest_data_com, btest_data_non):

    data1useruni_set = set(data1useruni_list)
    data2useruni_set = set(data2useruni_list)
    datausercom_set = set(datausercom_list)
    data1itemuni_set = set(data1itemuni_list)
    data2itemuni_set = set(data2itemuni_list)
    dataitemcom_set = set(dataitemcom_list)

    import copy
    dataitemcom_set_temp = copy.deepcopy(dataitemcom_set)
    data1itemchange = {}
    data2itemchange = {}
    changitemset = set()
    idxlen = len(dataitemcom_set_temp)
    iidfind = []
    for i in range(1, idxlen+1):
        iid = dataitemcom_set_temp.pop()
        changitemset.add(iid)
        data1itemchange[iid] = maxm+i*2
        data2itemchange[iid] = maxm+(i*2-1)
        iidfind.append([iid, maxm+i*2, maxm+(i*2-1)])
    maxm = maxm + idxlen*2

    def item_set_change(data1itemuni_set, data2itemuni_set, dataitemcom_set, iidfind):
        for i in range(len(iidfind)):
            dataitemcom_set.remove(iidfind[i][0])
            data1itemuni_set.add(iidfind[i][1])
            data2itemuni_set.add(iidfind[i][2])
        return data1itemuni_set, data2itemuni_set, dataitemcom_set
    data1itemuni_set, data2itemuni_set, dataitemcom_set = item_set_change(
        data1itemuni_set, data2itemuni_set, dataitemcom_set, iidfind)

    data1useruni_list = np.array(list(data1useruni_set), 'int32')
    data2useruni_list = np.array(list(data2useruni_set), 'int32')
    datausercom_list = np.array(list(datausercom_set), 'int32')
    data1itemuni_list = np.array(list(data1itemuni_set), 'int32')
    data2itemuni_list = np.array(list(data2itemuni_set), 'int32')
    dataitemcom_list = np.array(list(dataitemcom_set), 'int32')

    def traindata_item_change(train_data, itemchang, changitemset):

        for i in range(len(train_data[2])):
            if train_data[2][i] in changitemset:
                train_data[2][i] = itemchang[train_data[2][i]]
        return train_data

    atrain_data = traindata_item_change(
        atrain_data, data1itemchange, changitemset)
    btrain_data = traindata_item_change(
        btrain_data, data2itemchange, changitemset)

    def testdata_item_change(test_data, itemchang, changitemset):

        for i in range(len(test_data[1])):
            if test_data[1][i] in changitemset:
                test_data[1][i] = itemchang[test_data[1][i]]
        return test_data

    atest_data_uni = testdata_item_change(
        atest_data_uni, data1itemchange, changitemset)
    atest_data_com = testdata_item_change(
        atest_data_com, data1itemchange, changitemset)
    atest_data_non = testdata_item_change(
        atest_data_non, data1itemchange, changitemset)
    btest_data_uni = testdata_item_change(
        btest_data_uni, data2itemchange, changitemset)
    btest_data_com = testdata_item_change(
        btest_data_com, data2itemchange, changitemset)
    btest_data_non = testdata_item_change(
        btest_data_non, data2itemchange, changitemset)

    return iidfind, data1itemchange, data2itemchange, maxn, maxm, data1useruni_list, data2useruni_list, datausercom_list, data1itemuni_list, data2itemuni_list, dataitemcom_list, data1useruni_set, data2useruni_set, datausercom_set, data1itemuni_set, data2itemuni_set, dataitemcom_set, atrain_data, atest_data_uni, atest_data_com, atest_data_non, btrain_data, btest_data_uni, btest_data_com, btest_data_non


def read_data(data_str, dataset):

    if dataset[-4:] == 'OCCF':

        f1 = open(data_str)
        line = f1.readline().strip()
        data_list = []
        line = line.split(' ')
        data_list.append(list(map(int, line))[0:3])
        while line:
            line = f1.readline().strip()
            line = line.split(' ')
            if line == ['']:
                break
            data_list.append(list(map(int, line))[0:3])
        f1.close()

        return data_list
    else:

        f1 = open(data_str)
        line = f1.readline().strip()
        data_list = []
        line = line.split(' ')
        data_list.append(list(map(int, line))[0:4])
        while line:
            line = f1.readline().strip()
            line = line.split(' ')
            if line == ['']:
                break
            data_list.append(list(map(int, line))[0:3])
        f1.close()
        return data_list


def init_dataset(train_data_filename, test_data_filename_uni, test_data_filename_com, test_data_filename_non, dataset):
    train_data = read_data(train_data_filename, dataset)
    train_data = np.array(train_data, dtype='int32')
    maxn, maxm = max_n_m(train_data)
    logger.info(f"Maximum users {maxn} and items {maxm}")
    logger.info("Prepare training and testing data")
    train_data = trans(train_data)

    def test_data_init(test_data_filename_uni, dataset, maxn, maxm):
        test_data = read_data(test_data_filename_uni, dataset)
        test_data = np.array(test_data, dtype='int32')
        maxn, maxm = max_n_m(test_data, maxn, maxm)
        logger.info(f"Maximum users {maxn} and items {maxm}")
        logger.info("Prepare training and testing data")
        test_data = trans(test_data)
        testu, testi, testlabel = generate_test_data(test_data)
        test_data = [testu, testi, testlabel]
        return test_data, maxn, maxm

    test_data_uni, maxn, maxm = test_data_init(
        test_data_filename_uni, dataset, maxn, maxm)
    test_data_com, maxn, maxm = test_data_init(
        test_data_filename_com, dataset, maxn, maxm)
    test_data_non, maxn, maxm = test_data_init(
        test_data_filename_non, dataset, maxn, maxm)

    trainu, traini, trainlabel, train_user_index = generate_training_data(
        train_data, maxn, maxm)
    train_data = [train_user_index, trainu, traini, trainlabel]

    return train_data, test_data_uni, test_data_com, test_data_non, maxn, maxm


def model_init(maxn, maxm, hidden_dim=HIDDEN, dropout=DROP, lr=LR):
    model = MF(maxn, maxm, hidden_dim, dropout)

    optimizer = torch.optim.SGD(model.parameters(), lr)

    scheduler = 0
    if not ('cpu' in DEVICE):
        model = model.cuda()
    return model, optimizer, scheduler


if __name__ == "__main__":

    logger.info("Start loading data")
    atrain_data, atest_data_uni, atest_data_com, atest_data_non, maxna, maxma = init_dataset(
        atrain_data_filename, atest_data_filename_uni, atest_data_filename_com, atest_data_filename_non, dataset)
    btrain_data, btest_data_uni, btest_data_com, btest_data_non, maxnb, maxmb = init_dataset(
        btrain_data_filename, btest_data_filename_uni, btest_data_filename_com, btest_data_filename_non, dataset)

    maxn = max(maxna, maxnb)
    maxm = max(maxma, maxmb)

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

    if dataset == "ML100KCF":
        lr = 0.4
    elif dataset == "ML1MCF":
        lr = 0.3
    elif dataset == "NF5K5KCF":
        lr = 0.3

    SaveUV_dir = f'exp2-2/{dataset}-copy1-lr{lr}'
    iidfind = np.load(SaveUV_dir + "iidfind.npy")
    data1itemchange = {}
    data2itemchange = {}

    changitemset = set(iidfind[:, 0])
    for i in range(len(iidfind)):
        data1itemchange[iidfind[i][0]] = iidfind[i][1]
        data2itemchange[iidfind[i][0]] = iidfind[i][2]
    partner_list = ['a']
    maxm = maxm + len(dataitemcom_set)*2

    def item_set_change(data1itemuni_set, data2itemuni_set, dataitemcom_set, iidfind):
        for i in range(len(iidfind)):
            dataitemcom_set.remove(iidfind[i][0])
            data1itemuni_set.add(iidfind[i][1])
            data2itemuni_set.add(iidfind[i][2])
        return data1itemuni_set, data2itemuni_set, dataitemcom_set
    data1itemuni_set, data2itemuni_set, dataitemcom_set = item_set_change(
        data1itemuni_set, data2itemuni_set, dataitemcom_set, iidfind)

    def traindata_item_change(train_data, itemchang, changitemset):

        for i in range(len(train_data[2])):
            if train_data[2][i] in changitemset:
                train_data[2][i] = itemchang[train_data[2][i]]
        return train_data

    atrain_data = traindata_item_change(
        atrain_data, data1itemchange, changitemset)
    btrain_data = traindata_item_change(
        btrain_data, data2itemchange, changitemset)

    def testdata_item_change(test_data, itemchang, changitemset):

        for i in range(len(test_data[1])):
            if test_data[1][i] in changitemset:
                test_data[1][i] = itemchang[test_data[1][i]]
        return test_data

    atest_data_uni = testdata_item_change(
        atest_data_uni, data1itemchange, changitemset)
    atest_data_com = testdata_item_change(
        atest_data_com, data1itemchange, changitemset)
    atest_data_non = testdata_item_change(
        atest_data_non, data1itemchange, changitemset)
    btest_data_uni = testdata_item_change(
        btest_data_uni, data2itemchange, changitemset)
    btest_data_com = testdata_item_change(
        btest_data_com, data2itemchange, changitemset)
    btest_data_non = testdata_item_change(
        btest_data_non, data2itemchange, changitemset)

    data1useruni = indexlistto01vector(data1useruni_list, maxn)
    data2useruni = indexlistto01vector(data2useruni_list, maxn)
    datausercom = indexlistto01vector(datausercom_list, maxn)
    data1itemuni = indexlistto01vector(data1itemuni_list, maxm)
    data2itemuni = indexlistto01vector(data2itemuni_list, maxm)
    dataitemcom = indexlistto01vector(dataitemcom_list, maxm)

    SaveUV_dir_tempa = SaveUV_dir + f'-partnera-EP20-ep0'
    model_a = MF(maxn, maxm, HIDDEN, DROP)
    model_a.load_state_dict(torch.load(SaveUV_dir_tempa))
    model_a.to(DEVICE)
    model_a.eval()

    SaveUV_dir_tempb = SaveUV_dir + f'-partnerb-EP20-ep0'
    model_b = MF(maxn, maxm, HIDDEN, DROP)
    model_b.load_state_dict(torch.load(SaveUV_dir_tempb))
    model_b.to(DEVICE)
    model_b.eval()

    achange_item_id_set = set(iidfind[:, 1])

    uid_num_dict = {}
    uid_rating_dict = {}

    [atrain_user_index, atrainu, atraini, atrainlabel] = atrain_data
    for i in range(len(atrainu)):
        if atrainu[i] in datausercom_set and atraini[i] in achange_item_id_set:
            if atrainu[i] in uid_num_dict:
                uid_num_dict[atrainu[i]] = uid_num_dict[atrainu[i]] + 1
                uid_rating_dict[atrainu[i]].append(atraini[i])
            else:
                uid_num_dict[atrainu[i]] = 1
                uid_rating_dict[atrainu[i]] = []
                uid_rating_dict[atrainu[i]].append(atraini[i])

    del_list = []
    for key in uid_num_dict:
        if uid_num_dict[key] >= 5:
            choiced_uid_list = np.random.choice(
                uid_rating_dict[key], 5, replace=False)
            uid_rating_dict[key] = list(choiced_uid_list)
        else:
            del_list.append(key)

    for i in range(len(del_list)):
        uid_rating_dict.pop(del_list[i])

    atrain_data_dict = {}
    for i in range(len(atrainu)):
        atrain_data_dict[(atrainu[i], atraini[i])] = atrainlabel[i]

    while len(uid_rating_dict) > 100:
        uid_rating_dict.popitem()

    if attacktpye == "1":
        avg_sim_arrack_origin, avg_sim_correspond_origin, avg_sim_arrack_correspond, MAE, RMSE, std1, std2, std3 = attack_direct(
            uid_rating_dict, model_a, model_b, atrain_data_dict)
    elif attacktpye == "2":
        avg_sim_arrack_origin, avg_sim_correspond_origin, avg_sim_arrack_correspond, MAE, RMSE, std1, std2, std3 = attack_average(
            uid_rating_dict, model_a, model_b, atrain_data_dict)
    elif attacktpye == "3":
        scale_noise = scale_temp
        avg_sim_arrack_origin, avg_sim_correspond_origin, avg_sim_arrack_correspond, MAE, RMSE, std1, std2, std3 = attack_noise(
            uid_rating_dict, model_a, model_b, atrain_data_dict, scale_noise, clip_num)
    elif attacktpye == "4":
        scale_noise = scale_temp
        avg_sim_arrack_origin, avg_sim_correspond_origin, avg_sim_arrack_correspond, MAE, RMSE, std1, std2, std3 = attack_average_noise(
            uid_rating_dict, model_a, model_b, atrain_data_dict, scale_noise, clip_num)
    file_result = open(
        f'exp2-2/{dataset}-copy1-lr{lr}-attacktpye{attacktpye}.txt', 'w+')
    file_result.write(
        f'{avg_sim_arrack_origin,avg_sim_correspond_origin,avg_sim_arrack_correspond,MAE,RMSE,std1,std2,std3}')
    file_result.close()
 