import argparse
from Model import Model
from Dataset import Dataset
import pickle
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from utils import batch_collect, corpus_shift_right, valid
import torch.nn.functional as F

parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--bs', type=int, default=100, help='batch size')
parser.add_argument('--sustain', default='../model/e1s2800.pt', help='')
parser.add_argument('--toy', default=False, help='loads only 1000 training and 100 valid for testing')
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
if __name__ == '__main__':
    options = parser.parse_args()
    if options.toy:
        valid_dataset = Dataset(data_type='valid', length=options.bs)
        test_dataset = Dataset(data_type='test', length=options.bs)
    else:
        valid_dataset = Dataset(data_type='valid')
        test_dataset = Dataset(data_type='test')
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=options.bs, shuffle=False, collate_fn=batch_collect,
                              drop_last=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=options.bs, shuffle=False, collate_fn=batch_collect,
                             drop_last=True)
    model = Model(options)
    model = model.to(device)
    saved_state = torch.load(options.sustain)
    model.load_state_dict(saved_state)
    print('Already Load Pre Train Model')
    valid_loss, valid_ppl = valid(options, model, device, valid_loader)
    test_loss, test_ppl = valid(options, model, device, test_loader)

    # 设置所有的turn 精调后
    # Validation Average Loss: 4.528276        PPL: 92.525948
    # Validation Average Loss: 4.533181        PPL: 92.969895
