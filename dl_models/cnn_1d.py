#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2023/3/13 13:29
# @Author : LYX-夜光
from torch import nn

from dl_models import AD


class CNN_1D(AD):
    def __init__(self, learning_rate=0.001, epochs=100, batch_size=50, random_state=0, device='cpu', seq_len=60, sub_seq_len=30):
        super().__init__(learning_rate, epochs, batch_size, random_state, device, seq_len, sub_seq_len)
        self.model_name = "cnn_1d"

    def create_model(self):
        self.block1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=4),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.AvgPool1d(kernel_size=2, stride=2),
            nn.Dropout(p=0.05)
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.AvgPool1d(kernel_size=2, stride=2),
            nn.Dropout(p=0.05),
        )
        self.block3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.AvgPool1d(kernel_size=2, stride=2),
            nn.Dropout(p=0.05),
        )
        self.block4 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.AvgPool1d(kernel_size=2, stride=2),
            nn.Dropout(p=0.05),
        )
        self.dense = nn.Sequential(
            nn.Linear(in_features=1024, out_features=512),
            nn.Linear(in_features=512, out_features=self.label_num),
        )

    def forward(self, X):
        X = X.unsqueeze(1)
        X = self.block1(X)
        X = self.block2(X)
        X = self.block3(X)
        X = self.block4(X)
        y = self.dense(X.flatten(1))
        return y

