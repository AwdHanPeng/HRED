import torch.nn as nn, torch
import pickle
import numpy as np
from UtteranceEncoder import UtteranceEncoder
from Decoder import Decoder
from TurnEncoder import TurnEncoder
from Embedding import Embedding


class Model(nn.Module):
    def __init__(self, options):
        super(Model, self).__init__()
        self.Embedding = Embedding(options)
        self.UtteranceEncoder = UtteranceEncoder(options)
        self.TurnEncoder = TurnEncoder(options)
        self.Decoder = Decoder(options)

    def forward(self, utterances):
        '''


        :param utterances: [([],x),([],x),([],x)] also should consider batch

        [( (batch_size*max_length),(batch_size) ),......]

        :return:batch_size*max_length*total_words
        '''
        response = utterances[-1]
        context = utterances[:-1]
        hidden_states = []
        for utterance, utterance_length in context:
            hidden_states.append(self.UtteranceEncoder([self.Embedding(utterance), utterance_length]))
            # hidden_state: batch_size*hidden_state
        hidden_states = torch.stack(hidden_states, dim=1)  # hidden_state: batch_size*turns*hidden_state
        context_information = self.TurnEncoder(hidden_states)
        return self.Decoder(context_information, [self.Embedding(response[0]), response[1]], hidden_states[:, -1, :])
