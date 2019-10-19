import torch.nn as nn, torch
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self, options):
        super(Decoder, self).__init__()
        self.total_words = options.total_words
        self.dec_hidden_state = options.utter_hidden_state * options.utter_direction  # hidden state size of decoder rnn
        self.utter_hidden_state = options.utter_hidden_state  # hidden state size of utterance encoder rnn
        self.num_layers = 1
        self.dec_input_size = options.word_embedding_size + options.turn_hidden_state  # the input size of decoder rnn
        self.turn_hidden_size = options.turn_hidden_state  # the hidden state size of turn encoder rnn
        self.rnn = nn.GRU(hidden_size=self.dec_hidden_state, input_size=self.dec_input_size, num_layers=self.num_layers,
                          batch_first=True,
                          )
        self.tanh = nn.Tanh()
        self.generator_linear = nn.Linear(self.dec_hidden_state, self.total_words + 1)

    def forward(self, context_embedding, input, post_hidden_state):
        '''
        :param context_embedding: batch_size*turn_hidden_size
        :param input: [target_embedding,target_lengths]
        target_embedding: batch_size*max_length*embedding_size
        target_lengths: batch_size
                post_hidden_state: batch_size*utter_hidden_state
        :return:batch_size*max_length*total_words
        '''
        target_embedding, target_lengths = input[0], input[1]
        batch_size, max_sentence_length = target_embedding.shape[0], target_embedding.shape[1]

        target_embedding = torch.cat(
            (target_embedding, torch.unsqueeze(context_embedding, dim=1).repeat(1, max_sentence_length, 1)), dim=-1)

        pack_target_embedding = torch.nn.utils.rnn.pack_padded_sequence(target_embedding, target_lengths,
                                                                        batch_first=True,
                                                                        enforce_sorted=False)

        init_hidden_state = torch.unsqueeze(post_hidden_state, dim=0)
        hidden_states, last_hidden_state = self.rnn(pack_target_embedding, init_hidden_state)

        hidden_states, _ = torch.nn.utils.rnn.pad_packed_sequence(hidden_states, True)

        pro = self.generator_linear(hidden_states)
        return F.log_softmax(pro, dim=-1)
