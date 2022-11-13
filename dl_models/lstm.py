#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/10/24 15:17
# @Author : LYX-夜光

from torch import nn

from dl_models import AD


class LSTM_T_1(AD):
    def __init__(self, learning_rate=0.001, epochs=100, batch_size=50, random_state=0, device='cpu', seq_len=60, sub_seq_len=30):
        super().__init__(learning_rate, epochs, batch_size, random_state, device, seq_len, sub_seq_len)
        self.model_name = "lstm_t_1"

        self.input_size = 1

    def create_model(self):
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=32)
        self.linear = nn.Linear(in_features=32, out_features=self.label_num)

    def forward(self, X):
        X = X.unsqueeze(-1)
        X = X.permute(1, 0, 2)
        _, (h, _) = self.lstm(X)
        y = self.linear(h.squeeze(0))
        return y

class LSTM_F_1(LSTM_T_1):
    def __init__(self, learning_rate=0.001, epochs=100, batch_size=50, random_state=0, device='cpu', seq_len=60, sub_seq_len=30):
        super().__init__(learning_rate, epochs, batch_size, random_state, device, seq_len, sub_seq_len)
        self.model_name = "lstm_f_1"

class LSTM_T_2(LSTM_T_1):
    def __init__(self, learning_rate=0.001, epochs=100, batch_size=50, random_state=0, device='cpu', seq_len=60, sub_seq_len=30):
        super().__init__(learning_rate, epochs, batch_size, random_state, device, seq_len, sub_seq_len)
        self.model_name = "lstm_t_2"

        self.input_size = self.sub_seq_len

    def forward(self, X):
        X = X.permute(1, 0, 2)
        _, (h, _) = self.lstm(X)
        y = self.linear(h.squeeze(0))
        return y

class LSTM_F_2(LSTM_T_2):
    def __init__(self, learning_rate=0.001, epochs=100, batch_size=50, random_state=0, device='cpu', seq_len=60, sub_seq_len=30):
        super().__init__(learning_rate, epochs, batch_size, random_state, device, seq_len, sub_seq_len)
        self.model_name = "lstm_f_2" if self.sub_seq_len == 30 else "lstm_%d" % self.sub_seq_len

        self.input_size = int(self.sub_seq_len/2)+1
