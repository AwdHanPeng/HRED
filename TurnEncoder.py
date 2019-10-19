import torch.nn as nn, torch
import pickle
import numpy as np


class TurnEncoder(nn.Module):
    def __init__(self, options):
        super(TurnEncoder, self).__init__()
        self.hidden_state = options.turn_hidden_state
        self.input_size = options.utter_hidden_state * options.utter_direction

        self.rnn = nn.GRU(hidden_size=self.hidden_state, input_size=self.input_size, num_layers=1, bidirectional=False,
                          batch_first=True)

    def forward(self, input):
        '''

        :param input: batch_size*turns*hidden_state
        :return: batch_size*hidden_state
        '''
        hidden_states, last_hidden_state = self.rnn(input)
        last_hidden_state = last_hidden_state.squeeze(0)
        return last_hidden_state
