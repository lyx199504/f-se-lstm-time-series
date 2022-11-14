#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/10/24 17:40
# @Author : LYX-夜光

import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score

from dataPreprocessing import getDataset, standard
from dl_models.cnn_lstm_dnn import CNN_LSTM_DNN
from dl_models.lstm_dnn import LSTM_DNN
from dl_models.se_lstm_dnn import SE_LSTM_DNN

from optUtils import yaml_config
from optUtils.dataUtil import stratified_shuffle_index

import numpy.fft as nf


if __name__ == "__main__":
    seq_len = 60
    sub_seq_len = 30
    dataset_list = ['realTraffic', 'realKnownCause', 'realAWSCloudwatch', 'A1Benchmark']

    model_list = [LSTM_DNN, CNN_LSTM_DNN, SE_LSTM_DNN]

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
            model.device = 'cuda:3'
            model.metrics = f1_score
            model.metrics_list = [recall_score, precision_score, accuracy_score]
            model.fit(X[:train_point], y[:train_point], X[train_point:val_point], y[train_point:val_point])
            model.test_score(X[val_point:], y[val_point:])
