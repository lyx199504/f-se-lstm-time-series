#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/10/24 12:36
# @Author : LYX-夜光

import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score

from dataPreprocessing import getDataset, standard
from dl_models.cnn import CNN_T_1, CNN_F_1, CNN_T_2, CNN_F_2
from dl_models.dnn import DNN_T_1, DNN_F_1, DNN_T_2, DNN_F_2
from dl_models.lstm import LSTM_T_1, LSTM_F_1, LSTM_T_2, LSTM_F_2

from optUtils import yaml_config
from optUtils.dataUtil import stratified_shuffle_index

import numpy.fft as nf


if __name__ == "__main__":
    seq_len = 60
    sub_seq_len = 30
    dataset_list = ['realTraffic', 'realKnownCause', 'realAWSCloudwatch', 'A1Benchmark']

    model_list = [
        DNN_T_1, DNN_F_1, DNN_T_2, DNN_F_2,
        CNN_T_1, CNN_F_1, CNN_T_2, CNN_F_2,
        LSTM_T_1, LSTM_F_1, LSTM_T_2, LSTM_F_2,
    ]

    for model_clf in model_list:
        for dataset_name in dataset_list:
            X, y, r = getDataset(dataset_name, seq_len=seq_len, pre_list=[standard])

            seed, fold = yaml_config['cus_param']['seed'], yaml_config['cv_param']['fold']
            # 根据r的取值数量分层抽样
            shuffle_index = stratified_shuffle_index(r, n_splits=fold, random_state=seed)
            X, y = X[shuffle_index], y[shuffle_index]

            P, total = sum(y > 0), len(y)
            print("+: %d (%.2f%%)" % (P, P / total * 100), "-: %d (%.2f%%)" % (total - P, (1 - P / total) * 100))

            train_point, val_point = int(len(X) * 0.6), int(len(X) * 0.8)

            if model_clf in [DNN_F_1, CNN_F_1, LSTM_F_1]:
                X = np.abs(nf.rfft(X))

            if model_clf in [DNN_T_2, CNN_T_2, LSTM_T_2]:
                X_t_list = []
                for i in range(seq_len - sub_seq_len + 1):
                    X_t = X[:, i: i + sub_seq_len]
                    X_t_list.append(X_t)
                X = np.stack(X_t_list, axis=1)

            if model_clf in [DNN_F_2, CNN_F_2, LSTM_F_2]:
                # 子序列频率
                X_f_list = []
                for i in range(seq_len - sub_seq_len + 1):
                    X_f = np.abs(nf.rfft(X[:, i: i + sub_seq_len]))
                    X_f_list.append(X_f)
                X = np.stack(X_f_list, axis=1)

            model = model_clf(learning_rate=0.001, batch_size=512, epochs=500, random_state=1, seq_len=seq_len, sub_seq_len=sub_seq_len)
            model.model_name = model.model_name + "_%s" % dataset_name
            model.param_search = False
            model.save_model = True
            model.device = 'cuda:0'
            model.metrics = f1_score
            model.metrics_list = [recall_score, precision_score, accuracy_score]
            model.fit(X[:train_point], y[:train_point], X[train_point:val_point], y[train_point:val_point])
            model.test_score(X[val_point:], y[val_point:])
