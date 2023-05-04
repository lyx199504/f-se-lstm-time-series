#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2023/5/4 11:50
# @Author : LYX-夜光
from torch import nn

from dl_models import AD


class FFT_1D_CNN(AD):
    def __init__(self, learning_rate=0.001, epochs=100, batch_size=50, random_state=0, device='cpu', seq_len=60, sub_seq_len=30):
        super().__init__(learning_rate, epochs, batch_size, random_state, device, seq_len, sub_seq_len)
        self.model_name = "fft_1d_cnn"

    def create_model(self):
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 2, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        self.dense = nn.Sequential(
            nn.Linear(in_features=2*29, out_features=128),
            nn.Sigmoid(),
            nn.Linear(in_features=128, out_features=64),
        )
        self.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=64, out_features=2),
        )

    def forward(self, X):
        X = X.unsqueeze(1)
        X = self.cnn(X).flatten(1)
        X = self.dense(X)
        y = self.fc(X)
        return y


