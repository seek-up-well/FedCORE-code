import random
import numpy as np

from const import *
import math


def max_n_m(data, maxn=0, maxm=0):
    for i in range(len(data)):
        if data[i][0] > maxn:
            maxn = data[i][0]
        if data[i][1] > maxm:
            maxm = data[i][1]
    return int(maxn), int(maxm)


def max_n(data, maxn=0):
    for i in range(len(data)):
        if data[i][0] > maxn:
            maxn = data[i][0]
        if data[i][1] > maxn:
            maxn = data[i][1]
    return int(maxn)

 
def trans(data):
    for i in range(len(data)):
        data[i][0] = data[i][0]-1
        data[i][1] = data[i][1]-1
    return data


def test_acc(test_data_label, test_data_pred):
    mae = 0.0
    rmse = 0.0
    num = 0
    for i in range(len(test_data_label)):
        rating = test_data_label[i]
        pred = test_data_pred[i]
        if pred > 5:
            pred = 5
        if pred < 1:
            pred = 1

        err = pred - rating

        if np.isnan(err):
            num = num + 1
        else:
            mae = mae + abs(err)
            rmse = rmse + err*err

    MAE = round(mae / len(test_data_label), 4)
    RMSE = round(math.sqrt(rmse/len(test_data_label)), 4)

    return MAE, RMSE


def generate_training_data(train_data, maxn, maxm):

    trainu = []
    traini = []
    trainlabel = []
    train_user_index = {}
    indexs = 0

    user_rating_dict = {}
    user_index_dict = {}
    for i in range(maxn):
        user_index_dict[i] = []
        user_rating_dict[i] = set()
    for i in range(len(train_data)):
        trainu.append(train_data[i][0])
        traini.append(train_data[i][1])
        trainlabel.append(train_data[i][2])

        user_index_dict[train_data[i][0]].append(indexs)
        indexs = indexs + 1

        user_rating_dict[train_data[i][0]].add(train_data[i][1])

    for key in user_index_dict:
        if len(user_index_dict[key]):
            train_user_index[key] = user_index_dict[key]

    trainu = np.array(trainu, dtype='int32')
    traini = np.array(traini, dtype='int32')
    trainlabel = np.array(trainlabel, dtype='float32')
    return trainu, traini, trainlabel, train_user_index


def generate_test_data(test_data):

    testu = []
    testi = []
    testlabel = []
    for i in range(len(test_data)):
        testu.append(test_data[i][0])
        testi.append(test_data[i][1])
        testlabel.append(test_data[i][2])

    testu = np.array(testu, dtype='int32')
    testi = np.array(testi, dtype='int32')
    testlabel = np.array(testlabel, dtype='float32')
    return testu, testi, testlabel
