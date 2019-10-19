import torch.nn as nn, torch
import pickle
import numpy as np
from tqdm import tqdm


class Embedding(nn.Module):
    def __init__(self, options):
        super(Embedding, self).__init__()
        self.word_embedding_size = options.word_embedding_size
        self.total_words = options.total_words
        self.word_embedding = nn.Embedding(self.total_words + 1, self.word_embedding_size, padding_idx=self.total_words,
                                           sparse=False)
        self.dropout = nn.Dropout(options.dropout)
        self.embedding = options.embedding

        if self.embedding is True and options.sustain is None:
            print('Load Pre-Train Embedding!')
            embedding_matrix = self.load_embedding()
            embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float32)
            self.word_embedding.weight = torch.nn.Parameter(embedding_matrix)

    def forward(self, input):
        return self.dropout(self.word_embedding(input))

    def load_embedding(self):
        vocab_file = '../Data/Dataset.dict.pkl'
        embed_file = '../embedding/glove.840B.300d.txt'
        vocab = {}
        print('Load Vocab File')
        with open(vocab_file, 'rb') as f:
            dict_data = pickle.load(f)
        for t in tqdm(dict_data):
            token, index, _, _ = t
            vocab[token] = index
        print('Load Raw Embedding File')
        embedding_index = {}
        with open(embed_file, 'r', encoding='UTF-8') as f:
            lines = f.readlines()
            for line in tqdm(lines):
                values = line.split()
                word = values[0]
                if len(values) > 301:
                    continue
                embedding_vector = np.asarray(values[1:], dtype='float32')
                embedding_index[word] = embedding_vector
        embedding_matrix = np.zeros((self.total_words + 1, self.word_embedding_size))
        print('Create Embedding Matrix')
        count = 0
        for word, i in tqdm(vocab.items()):
            embedding_vector = embedding_index.get(word)
            if embedding_vector is not None:
                count += 1
                assert len(embedding_vector) == self.word_embedding_size
                # 指 从预训练embedding表中 查找出需要使用的 embedding vector
                embedding_matrix[i] = embedding_vector
        print('Find Embedding Count {}'.format(count))
        return embedding_matrix
