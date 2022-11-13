#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/9/6 16:02
# @Author : LYX-夜光
from torch import nn

from dl_models import AD


class C_LSTM(AD):
    def __init__(self, learning_rate=0.001, epochs=100, batch_size=50, random_state=0, device='cpu', seq_len=60, sub_seq_len=30):
        super().__init__(learning_rate, epochs, batch_size, random_state, device, seq_len, sub_seq_len)
        self.model_name = "c_lstm"

    def create_model(self):
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=5, stride=1, padding=2),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.lstm = nn.LSTM(input_size=64, hidden_size=64)
        self.dnn = nn.Sequential(
            nn.Linear(in_features=64, out_features=32),
            nn.Tanh(),
            nn.Linear(in_features=32, out_features=self.label_num),
        )

    def forward(self, X):
        X = X.unsqueeze(1)
        X = self.conv1(X)
        X = self.conv2(X)
        X = X.permute(2, 0, 1)
        _, (h, _) = self.lstm(X)
        y = self.dnn(h.squeeze(0))
        return y
