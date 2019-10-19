import torch.nn as nn, torch
import pickle
import numpy as np


class UtteranceEncoder(nn.Module):
    def __init__(self, options):
        super(UtteranceEncoder, self).__init__()
        self.word_embedding_size = options.word_embedding_size
        self.hidden_state = options.utter_hidden_state

        self.num_layers = options.utter_num_layers

        self.direction = options.utter_direction
        self.bidirection = False if options.utter_direction == 1 else True
        self.rnn = nn.GRU(input_size=self.word_embedding_size, hidden_size=self.hidden_state,
                          num_layers=self.num_layers, bidirectional=self.bidirection, batch_first=True,
                          dropout=options.dropout)

    def forward(self, input):
        '''

        :param input: [utterance_embedding,utterance_length]
        utterance_embedding: batch_size*max_length*embedding_size
        utterance_length: batch_size
        :return:batch_size*hidden_state
        '''
        utterance_embedding, utterance_length = input[0], input[1]  # utterance_length is real length
        batch_size, seq_len = utterance_embedding.shape[0], utterance_embedding.shape[1]
        utterance_embedding = torch.nn.utils.rnn.pack_padded_sequence(utterance_embedding, utterance_length,
                                                                      enforce_sorted=False, batch_first=True)
        hidden_states, last_hidden_states = self.rnn(utterance_embedding)  # this is so cool to use pack
        last_hidden_states = last_hidden_states.view(self.num_layers, self.direction, batch_size, self.hidden_state)
        last_hidden_states = last_hidden_states[-1, :, :, :].squeeze(0)
        last_hidden_states = last_hidden_states.permute(1, 0, 2)
        last_hidden_states = last_hidden_states.reshape(batch_size, self.hidden_state * self.direction)
        return last_hidden_states
