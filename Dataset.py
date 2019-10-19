import torch
import copy
import pickle
from torch.utils.data import Dataset as DS
from tqdm import tqdm
import random

__eot__ = 1
max_sentence_length = 100


def dialogue(line):
    temp_utterance = [__eot__, ]
    dialogue = []
    for item in line:
        if item is not __eot__:
            temp_utterance.append(item)
        else:
            while len(temp_utterance) > max_sentence_length:
                temp_utterance.pop()
            dialogue.append({
                'utterance': copy.deepcopy(temp_utterance),
                'length': len(temp_utterance)
            })
            temp_utterance = [__eot__, ]
    while len(temp_utterance) > max_sentence_length:
        temp_utterance.pop()
    dialogue.append({
        'utterance': copy.deepcopy(temp_utterance),
        'length': len(temp_utterance)
    })
    return {
        'dialogue': dialogue,
        'turns': len(dialogue),
        'tokens': len(line)
    }


class Dataset(DS):
    def __init__(self, data_type, length=None):
        if data_type == 'train':
            _file = '../Data/Training.dialogues.pkl'
        elif data_type == 'valid':
            _file = '../Data/Validation.dialogues.pkl'
        elif data_type == 'test':
            _file = '../Data/Test.dialogues.pkl'
        self.data = []
        print('It is time to Load {} Data'.format(data_type))
        with open(_file, 'rb') as f:
            data = pickle.load(f)
            if length is not None:
                border = random.randint(0, 10000)
                for i, line in enumerate(data):
                    if border <= i < border + length:
                        self.data.append(dialogue(line))
                    elif i >= border + length:
                        break
            else:
                for line in tqdm(data):
                    self.data.append(dialogue(line))
        print('Already Load {} Data: {}'.format(data_type, len(self.data)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
