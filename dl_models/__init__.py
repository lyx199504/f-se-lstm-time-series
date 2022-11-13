#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/4/23 22:35
# @Author : LYX-夜光

import joblib

from optUtils import yaml_config
from optUtils.logUtil import logging_config, get_lines_from_log
from optUtils.pytorchModel import DeepLearningClassifier


class AD(DeepLearningClassifier):
    def __init__(self, learning_rate=0.001, epochs=100, batch_size=50, random_state=0, device='cpu', seq_len=60, sub_seq_len=30):
        super().__init__(learning_rate, epochs, batch_size, random_state, device)
        self.label_num = 2  # 二分类
        self.seq_len = seq_len
        self.sub_seq_len = sub_seq_len

    # 测试数据
    def test_score(self, X, y):
        model_param_list = get_lines_from_log(self.model_name, (0, self.epochs))
        val_score_list = list(map(lambda x: x['best_score_'], model_param_list))
        length = 11
        val_score = sum(val_score_list[:length])
        best_val_score = val_score
        epoch = int(length/2)
        for i in range(length, len(val_score_list)):
            val_score = val_score - val_score_list[i-length] + val_score_list[i]
            if best_val_score < val_score:
                best_val_score = val_score
                epoch = i - int((length-1)/2)
        model_param = model_param_list[epoch]
        log_dir = yaml_config['dir']['log_dir']
        logger = logging_config(self.model_name, log_dir + '/%s.log' % self.model_name)
        logger.info({
            "===================== Test score of the selected model ====================="
        })
        mdl = joblib.load(model_param['model_path'])
        mdl.device = self.device
        test_score = mdl.score(X, y, batch_size=512)
        test_score_list = mdl.score_list(X, y, batch_size=512)
        test_score_dict = {self.metrics.__name__: test_score}
        for i, metrics in enumerate(self.metrics_list):
            test_score_dict.update({metrics.__name__: test_score_list[i]})

        logger.info({
            "select_epoch": model_param['epoch'],
            "test_score_dict": test_score_dict,
        })
