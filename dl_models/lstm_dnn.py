#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/4/26 15:07
# @Author : LYX-夜光

from torch import nn

from dl_models import AD


class LSTM_DNN(AD):
    def __init__(self, learning_rate=0.001, epochs=100, batch_size=50, random_state=0, device='cpu', seq_len=60, sub_seq_len=30):
        super().__init__(learning_rate, epochs, batch_size, random_state, device, seq_len, sub_seq_len)
        self.model_name = "lstm_dnn"

    def create_model(self):
        self.lstm = nn.LSTM(input_size=int(self.sub_seq_len/2)+1, hidden_size=64)
        self.dnn = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=64, out_features=32),
            nn.Tanh(),
            nn.Linear(in_features=32, out_features=self.label_num),
        )

    def forward(self, X):
        X = X.permute(1, 0, 2)
        _, (h, _) = self.lstm(X)
        y = self.dnn(h.squeeze(0))
        return y
