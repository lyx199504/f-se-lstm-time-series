#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/10/22 23:58
# @Author : LYX-夜光

import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score

from dataPreprocessing import getDataset, standard

from optUtils import yaml_config
from optUtils.dataUtil import stratified_shuffle_index

import numpy.fft as nf

from optUtils.trainUtil import ml_train


if __name__ == "__main__":
    seq_len = 60
    dataset_list = ['realTraffic', 'realKnownCause', 'realAWSCloudwatch', 'A1Benchmark']
    model_list = ['knn_clf', 'svm_clf', 'lr_clf', 'dt_clf', 'rf_clf']

    for dataset_name in dataset_list:
        X, y, r = getDataset(dataset_name, seq_len=seq_len, pre_list=[standard])

        seed, fold = yaml_config['cus_param']['seed'], yaml_config['cv_param']['fold']
        # 根据r的取值数量分层抽样
        shuffle_index = stratified_shuffle_index(r, n_splits=fold, random_state=seed)
        X, y = X[shuffle_index], y[shuffle_index]

        P, total = sum(y > 0), len(y)
        print("+: %d (%.2f%%)" % (P, P / total * 100), "-: %d (%.2f%%)" % (total - P, (1 - P / total) * 100))

        X_f = np.abs(nf.rfft(X))

        train_point = int(len(X) * 0.6)

        for model_name in model_list:
            ml_train(
                X[:train_point], y[:train_point],
                X[train_point:], y[train_point:],
                model_name=model_name,
                metrics_list=[f1_score, recall_score, precision_score, accuracy_score],
                note=dataset_name + "_t",
            )

            ml_train(
                X_f[:train_point], y[:train_point],
                X_f[train_point:], y[train_point:],
                model_name=model_name,
                metrics_list=[f1_score, recall_score, precision_score, accuracy_score],
                note=dataset_name + "_f",
            )



