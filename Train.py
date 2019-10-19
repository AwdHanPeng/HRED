import argparse
from Model import Model
from Dataset import Dataset
import pickle
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from utils import batch_collect, corpus_shift_right, valid
import torch.nn.functional as F

embedding_file = ''
parser = argparse.ArgumentParser(description='HRED Parameter')

parser.add_argument('--word-embedding-size', type=int, default=300)
parser.add_argument('--utter-hidden-state', type=int, default=600, help='encoder utterance hidden state')
parser.add_argument('--utter-num-layers', type=int, default=2, help='')
parser.add_argument('--utter-direction', type=int, default=2, help='')
parser.add_argument('--total-words', type=int, default=20000, help='')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout probability used all throughout')
parser.add_argument('--turn-hidden-state', type=int, default=1200, help='encoder session hidden state')
parser.add_argument('--embedding', default=True, help='')
parser.add_argument('--lr', type=float, default=0.00005, help='learning rate for optimizer')  # 这个很关键！！
parser.add_argument('--bs', type=int, default=100, help='batch size')
parser.add_argument('--epoch', type=int, default=10, help='number of epochs')
parser.add_argument('--toy', default=False, help='loads only 1000 training and 100 valid for testing')
parser.add_argument('--sustain', default='../model/e1s750.pt', help='')
parser.add_argument('--log-interval', type=int, default=1,
                    help='how many batches to wait before logging training status')
parser.add_argument('--valid-interval', type=int, default=10,
                    help='how many batches to wait before logging training status')
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

if __name__ == '__main__':
    history_list = []
    options = parser.parse_args()
    if options.toy:
        train_dataset = Dataset(data_type='train', length=1000)
    else:
        train_dataset = Dataset(data_type='train')
    train_loader = DataLoader(dataset=train_dataset, batch_size=options.bs, shuffle=True, collate_fn=batch_collect,
                              drop_last=True)
    best_loss = float('Inf')
    model = Model(options)
    model = model.to(device)
    if options.sustain is not None:
        saved_state = torch.load(options.sustain)
        model.load_state_dict(saved_state)
        print('Already Load Pre Train Model')
    optimizer = optim.Adam(model.parameters(), options.lr)
    print(model.parameters)
    for epoch in range(1, options.epoch + 1):
        for batch_idx, origin_corpus in enumerate(train_loader):
            model.train()
            response_y = corpus_shift_right(origin_corpus, options.bs)
            for i, (utterance, length) in enumerate(origin_corpus):
                origin_corpus[i] = (utterance.to(device), length.to(device))  # to(device)不是原地操作
            response_y = response_y.to(device)
            prob = model(origin_corpus)
            loss = F.nll_loss(prob.view(-1, options.total_words + 1), response_y.view(-1),
                              ignore_index=options.total_words, reduction='mean')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)  # so important!!!!!!!!!!
            optimizer.step()
            if batch_idx % options.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t'.format(
                    epoch, batch_idx * options.bs, len(train_loader.dataset), 100.0 * batch_idx / len(train_loader),
                    loss))
            # -------------------------------------------------------
            if batch_idx % options.valid_interval == 0:
                valid_dataset = Dataset(data_type='valid', length=1000)
                valid_loader = DataLoader(dataset=valid_dataset, batch_size=options.bs, shuffle=False,
                                          collate_fn=batch_collect, drop_last=True)
                valid_loss, ppl = valid(options, model, device, valid_loader)
                history_list.append({
                    'epoch': epoch,
                    'batch_idx': batch_idx,
                    'loss': valid_loss,
                    'ppl': ppl
                })
                if valid_loss <= best_loss:
                    best_epo = epoch
                    best_loss = valid_loss
                    best_step = batch_idx
                    torch.save(model.state_dict(), "../model/e{}s{}.pt".format(best_epo, best_step))
                    print('save model when epoch = {} step = {} as e{}s{}.pt'.format(best_epo, best_step, best_epo,
                                                                                     best_step))
    with open('../model/history.pkl', 'wb') as f:
        pickle.dump(history_list, f)
