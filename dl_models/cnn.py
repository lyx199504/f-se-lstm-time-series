#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/10/24 13:39
# @Author : LYX-夜光

from torch import nn

from dl_models import AD


class CNN_T_1(AD):
    def __init__(self, learning_rate=0.001, epochs=100, batch_size=50, random_state=0, device='cpu', seq_len=60, sub_seq_len=30):
        super().__init__(learning_rate, epochs, batch_size, random_state, device, seq_len, sub_seq_len)
        self.model_name = "cnn_t_1"

        self.output_size = int((self.seq_len-3)/3)

    def create_model(self):
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=4, stride=1),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=3, stride=3),
        )
        self.linear = nn.Linear(in_features=16*self.output_size, out_features=self.label_num)

    def forward(self, X):
        X = X.unsqueeze(1)
        X = self.cnn(X)
        X = X.flatten(1)
        y = self.linear(X)
        return y

class CNN_F_1(CNN_T_1):
    def __init__(self, learning_rate=0.001, epochs=100, batch_size=50, random_state=0, device='cpu', seq_len=60, sub_seq_len=30):
        super().__init__(learning_rate, epochs, batch_size, random_state, device, seq_len, sub_seq_len)
        self.model_name = "cnn_f_1"

        self.output_size = int((int(self.seq_len/2)+1-3)/3)

class CNN_T_2(AD):
    def __init__(self, learning_rate=0.001, epochs=100, batch_size=50, random_state=0, device='cpu', seq_len=60, sub_seq_len=30):
        super().__init__(learning_rate, epochs, batch_size, random_state, device, seq_len, sub_seq_len)
        self.model_name = "cnn_t_2"

        self.output_size = int((self.sub_seq_len-3)/3)*int((self.seq_len-self.sub_seq_len+1-3)/3)

    def create_model(self):
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(4, 4), stride=(1, 1)),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3)),
        )
        self.linear = nn.Linear(in_features=16*self.output_size, out_features=self.label_num)

    def forward(self, X):
        X = X.unsqueeze(1)
        X = self.cnn(X)
        X = X.flatten(1)
        y = self.linear(X)
        return y

class CNN_F_2(CNN_T_2):
    def __init__(self, learning_rate=0.001, epochs=100, batch_size=50, random_state=0, device='cpu', seq_len=60, sub_seq_len=30):
        super().__init__(learning_rate, epochs, batch_size, random_state, device, seq_len, sub_seq_len)
        self.model_name = "cnn_f_2" if self.sub_seq_len == 30 else "cnn_%d" % self.sub_seq_len

        self.output_size = int((int(self.sub_seq_len/2)+1-3)/3)*int((self.seq_len-self.sub_seq_len+1-3)/3)
