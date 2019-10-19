import argparse
from Model import Model
from Dataset import Dataset
import pickle
import torch
from torch.utils.data import DataLoader
from utils import batch_collect, infer
from tqdm import tqdm

__eot__ = 1

parser = argparse.ArgumentParser(description='Infer')
parser.add_argument('--bs', type=int, default=1, help='batch size')
parser.add_argument('--sustain', default='../model/e1s2800.pt', help='')
parser.add_argument('--toy', default=True, help='loads only 1000 training and 100 valid for testing')
parser.add_argument('--word-embedding-size', type=int, default=300)
parser.add_argument('--utter-hidden-state', type=int, default=600, help='encoder utterance hidden state')
parser.add_argument('--utter-num-layers', type=int, default=2, help='')
parser.add_argument('--utter-direction', type=int, default=2, help='')
parser.add_argument('--total-words', type=int, default=20000, help='')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout probability used all throughout')
parser.add_argument('--turn-hidden-state', type=int, default=1200, help='encoder session hidden state')
parser.add_argument('--embedding', default=True, help='')
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
vocab_file = '../Data/Dataset.dict.pkl'

if __name__ == '__main__':
    options = parser.parse_args()
    if options.toy:
        test_dataset = Dataset(data_type='test', length=100)
    else:
        test_dataset = Dataset(data_type='test')
    test_loader = DataLoader(dataset=test_dataset, batch_size=options.bs, shuffle=False, collate_fn=batch_collect,
                             drop_last=True)
    model = Model(options)
    model = model.to(device)
    saved_state = torch.load(options.sustain, map_location=device)
    model.load_state_dict(saved_state)
    print('Already Load Pre Train Model')
    dic = {}
    print('Load Vocab File')
    with open(vocab_file, 'rb') as f:
        dict_data = pickle.load(f)
        for t in tqdm(dict_data):
            token, index, _, _ = t
            dic[index] = token
    dic[options.total_words] = 'PAD'
    generation = infer(options, model, device, test_loader, 5, dic)
    print('It is time to save generation sentences')
    with open('../data/generation.pkl', 'wb') as f:
        pickle.dump(generation, f)
