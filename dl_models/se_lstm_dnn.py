#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/5/15 21:44
# @Author : LYX-夜光

from torch import nn

from dl_models import AD

class SE(nn.Module):
    def __init__(self, in_channels, ratio):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # squeeze
            nn.Conv2d(in_channels, in_channels//ratio, kernel_size=(1, 1)),  # compress
            nn.Tanh(),
            nn.Conv2d(in_channels//ratio, in_channels, kernel_size=(1, 1)),  # excitation
            nn.Sigmoid(),
        )

    def forward(self, X):
        return self.se(X)

class SENet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, ratio):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.se = SE(out_channels, ratio)
        self.tanh = nn.Tanh()

    def forward(self, X):
        X = self.conv(X)
        coef = self.se(X)
        X = X * coef
        return self.tanh(X)

class SE_LSTM_DNN(AD):
    def __init__(self, learning_rate=0.001, epochs=100, batch_size=50, random_state=0, device='cpu', seq_len=60, sub_seq_len=30):
        super().__init__(learning_rate, epochs, batch_size, random_state, device, seq_len, sub_seq_len)
        self.model_name = "se_lstm_dnn" if sub_seq_len == 30 else "se_lstm_dnn_%d" % sub_seq_len

        self.input_size = int((int(sub_seq_len/2)+1)/4)

    def create_model(self):
        self.senet = SENet(1, 16, (1, 4), (1, 4), 4)
        self.lstm = nn.LSTM(input_size=16*self.input_size, hidden_size=64)
        self.dnn = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=64, out_features=32),
            nn.Tanh(),
            nn.Linear(in_features=32, out_features=self.label_num),
        )

    def forward(self, X):
        X = X.unsqueeze(1)
        X = self.senet(X)
        H = X.permute(2, 0, 1, 3).flatten(2)
        _, (h, _) = self.lstm(H)
        y = self.dnn(h.squeeze(0))
        return y
