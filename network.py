import math

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
import logging


def orthogonal_init(layer, gain=np.sqrt(2)):
    for name, param in layer.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0)
        elif 'weight' in name:
            nn.init.orthogonal_(param, gain=gain)


class Base_LSTM(nn.Module):
    def reset_hidden_state(self, device, batch_size=None):
        if batch_size is None:
            self.hidden_state = torch.zeros(self.lstm.num_layers, self.lstm.hidden_size).to(device)
            self.cell_state = torch.zeros(self.lstm.num_layers, self.lstm.hidden_size).to(device)
        else:
            self.hidden_state = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(device)
            self.cell_state = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(device)

    def forward(self, s):
        raise NotImplementedError


class Actor_LSTM(Base_LSTM):
    def __init__(self, args):
        super(Actor_LSTM, self).__init__()

        self.lstm = nn.LSTM(input_size=args.state_dim, hidden_size=args.lstm_hidden_dim,
                            num_layers=args.lstm_num_layers, batch_first=True)

        self.mean_layer = nn.Linear(args.lstm_hidden_dim, args.action_dim)

        if args.std:
            self.log_std = nn.Parameter(torch.tensor(args.std).log(), requires_grad=False)
        else:
            self.log_std = nn.Parameter(torch.zeros(1, args.action_dim))

        if args.use_orthogonal_init:
            logging.info("------use_orthogonal_init------")
            if hasattr(self, 'mean_layer'):
                orthogonal_init(self.mean_layer, gain=0.01)

            if hasattr(self, 'std_layer'):
                orthogonal_init(self.std_layer, gain=0.01)

    def forward(self, s):
        # s: [batch_size, seq_len, hidden_dim]

        s, (self.hidden_state, self.cell_state) = self.lstm(s, (self.hidden_state, self.cell_state))
        # s: [batch_size, seq_len, hidden_size_lstm]

        mean = torch.tanh(self.mean_layer(s))
        # mean: [batch_size, seq_len, action_dim]

        return mean, self.log_std.expand_as(mean).exp()

    def pdf(self, s):
        mean, std = self.forward(s)
        return Normal(mean, std)


class Critic_LSTM(Base_LSTM):
    def __init__(self, args):
        super(Critic_LSTM, self).__init__()

        self.lstm = nn.LSTM(input_size=args.state_dim, hidden_size=args.lstm_hidden_dim,
                            num_layers=args.lstm_num_layers, batch_first=True)

        self.value_layer = nn.Linear(args.lstm_hidden_dim, 1)

    def forward(self, s):
        # s: [batch_size, seq_len, hidden_dim]

        s, (self.hidden_state, self.cell_state) = self.lstm(s, (self.hidden_state, self.cell_state))
        # s: [batch_size, seq_len, hidden_size_lstm]

        value = self.value_layer(s)
        # mean: [batch_size, seq_len, action_dim]

        return value
