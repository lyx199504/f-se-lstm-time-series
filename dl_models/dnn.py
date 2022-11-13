#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/10/24 12:39
# @Author : LYX-夜光
from torch import nn

from dl_models import AD


class DNN_T_1(AD):
    def __init__(self, learning_rate=0.001, epochs=100, batch_size=50, random_state=0, device='cpu', seq_len=60, sub_seq_len=30):
        super().__init__(learning_rate, epochs, batch_size, random_state, device, seq_len, sub_seq_len)
        self.model_name = "dnn_t_1"

        self.input_size = seq_len

    def create_model(self):
        self.dnn = nn.Sequential(
            nn.Linear(in_features=self.input_size, out_features=64),
            nn.Tanh(),
            nn.Linear(in_features=64, out_features=32),
            nn.Tanh(),
            nn.Linear(in_features=32, out_features=self.label_num),
        )

    def forward(self, X):
        return self.dnn(X)

class DNN_F_1(DNN_T_1):
    def __init__(self, learning_rate=0.001, epochs=100, batch_size=50, random_state=0, device='cpu', seq_len=60, sub_seq_len=30):
        super().__init__(learning_rate, epochs, batch_size, random_state, device, seq_len, sub_seq_len)
        self.model_name = "dnn_f_1"

        self.input_size = int(self.seq_len/2)+1

class DNN_T_2(AD):
    def __init__(self, learning_rate=0.001, epochs=100, batch_size=50, random_state=0, device='cpu', seq_len=60, sub_seq_len=30):
        super().__init__(learning_rate, epochs, batch_size, random_state, device, seq_len, sub_seq_len)
        self.model_name = "dnn_t_2"

        self.input_size = self.sub_seq_len
        self.time_step = self.seq_len - self.sub_seq_len + 1

    def create_model(self):
        self.dnn1 = nn.Sequential(
            nn.Linear(in_features=self.input_size, out_features=32),
            nn.Tanh(),
            nn.Linear(in_features=32, out_features=2),
        )
        self.dnn2 = nn.Sequential(
            nn.Linear(in_features=2*self.time_step, out_features=32),
            nn.Tanh(),
            nn.Linear(in_features=32, out_features=self.label_num),
        )

    def forward(self, X):
        X = self.dnn1(X)
        X = X.flatten(1)
        y = self.dnn2(X)
        return y

class DNN_F_2(DNN_T_2):
    def __init__(self, learning_rate=0.001, epochs=100, batch_size=50, random_state=0, device='cpu', seq_len=60, sub_seq_len=30):
        super().__init__(learning_rate, epochs, batch_size, random_state, device, seq_len, sub_seq_len)
        self.model_name = "dnn_f_2" if self.sub_seq_len == 30 else "dnn_%d" % self.sub_seq_len

        self.input_size = int(self.sub_seq_len/2)+1
