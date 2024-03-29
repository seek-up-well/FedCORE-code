import random
import numpy as np
from loguru import logger


def sub_sample(points, num):

    ind = np.arange(len(points))
    sub_ind = np.random.choice(ind, num, replace=False)
    sub_points = np.array(points)[sub_ind]
    return sub_points


def batch_data_select(trainu, traini, trainlabel, batch_size):
    indexs = np.array(range(0, len(trainu)), dtype='int32')
    choice_indexs = np.random.choice(indexs, batch_size, replace=False)
    uid = trainu[choice_indexs]
    iid = traini[choice_indexs]
    y = trainlabel[choice_indexs]

    uid = np.expand_dims(uid, axis=1)

    iid = np.expand_dims(iid, axis=1)
    return uid, iid, y


def generate_batch_data_randomCo(batch_size, atrain_data, btrain_data, iter_num):

    train_user_index_a, trainu_a, traini_a, trainlabel_a = atrain_data
    train_user_index_b, trainu_b, traini_b, trainlabel_b = btrain_data

    for _ in range(iter_num):

        uid_a, iid_a, y_a = batch_data_select(
            trainu_a, traini_a, trainlabel_a, batch_size)
        uid_b, iid_b, y_b = batch_data_select(
            trainu_b, traini_b, trainlabel_b, batch_size)

        yield ([uid_a, iid_a], [y_a], [uid_b, iid_b], [y_b])


def generate_batch_data_random(batch_size, train_user_index, trainu, traini, trainlabel):

    iter_num = len(trainu)//batch_size
    for _ in range(iter_num):

        indexs = np.array(range(0, len(trainu)), dtype='int32')
        choice_indexs = np.random.choice(indexs, batch_size, replace=False)
        uid = trainu[choice_indexs]
        iid = traini[choice_indexs]
        y = trainlabel[choice_indexs]
        uid = np.expand_dims(uid, axis=1)
        iid = np.expand_dims(iid, axis=1)

        yield ([uid, iid], [y])


def wipe_test_data(train_user_set, train_item_set, test_data):
    test_data_temp = []
    for i in range(len(test_data)):
        if (test_data[i][0] in train_user_set) and (test_data[i][1] in train_item_set):
            test_data_temp.append(
                [test_data[i][0], test_data[i][1], test_data[i][2]]) 

    test_data_temp = np.array(test_data_temp, dtype='int32')
    return test_data_temp
