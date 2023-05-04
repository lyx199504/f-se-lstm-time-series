#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/10/24 20:13
# @Author : LYX-夜光

import numpy as np
import numpy.fft as nf
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score

from dataPreprocessing import getDataset, standard, norm
from dl_models.c_lstm import C_LSTM
from dl_models.c_lstm_ae import C_LSTM_AE
from dl_models.cnn_1d import CNN_1D
from dl_models.fft_1d_cnn import FFT_1D_CNN
from dl_models.tcn import TCN

from optUtils import yaml_config
from optUtils.dataUtil import stratified_shuffle_index


if __name__ == "__main__":
    seq_len = 60
    dataset_list = ['realTraffic', 'realKnownCause', 'realAWSCloudwatch', 'A1Benchmark']
    model_list = [C_LSTM, C_LSTM_AE, CNN_1D, TCN, FFT_1D_CNN]

    for model_clf in model_list:
        for dataset_name in dataset_list:
            X, y, r = getDataset(dataset_name, seq_len=seq_len, pre_list=[standard])

            seed, fold = yaml_config['cus_param']['seed'], yaml_config['cv_param']['fold']
            # 根据r的取值数量分层抽样
            shuffle_index = stratified_shuffle_index(r, n_splits=fold, random_state=seed)
            X, y = X[shuffle_index], y[shuffle_index]

            if model_clf == C_LSTM_AE:
                length = 10
                X = np.array([[x[i: i + length] for i in range(len(x) - length + 1)] for x in X])

            if model_clf == FFT_1D_CNN:
                X = np.abs(nf.fft(X))

            P, total = sum(y > 0), len(y)
            print("+: %d (%.2f%%)" % (P, P / total * 100), "-: %d (%.2f%%)" % (total - P, (1 - P / total) * 100))

            train_point, val_point = int(len(X) * 0.6), int(len(X) * 0.8)

            model = model_clf(learning_rate=0.001, batch_size=512, epochs=500, random_state=1, seq_len=seq_len)

            # model.create_model()
            # print(sum([param.nelement() for param in model.parameters()]))
            # exit()

            model.model_name = model.model_name + "_%s" % dataset_name
            model.param_search = False
            model.save_model = True
            model.device = 'cuda:0'
            model.metrics = f1_score
            model.metrics_list = [recall_score, precision_score, accuracy_score]
            model.fit(X[:train_point], y[:train_point], X[train_point:val_point], y[train_point:val_point])
            model.test_score(X[val_point:], y[val_point:])
