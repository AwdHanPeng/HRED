import torch
import copy
import pickle
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import math

__eot__ = 1
total_words = 20000
max_turns = 100
max_sentence_length = 100
__eou__ = 18575


def batch_collect(batch):
    '''
    :param batch:
    [( (batch_size*max_length),(batch_size) ),......] 每个tuple是一轮
    :return:
    '''
    # 首先第一步确定max turn 然后填充
    batch = copy.deepcopy(batch)  # 这个地方太重要了啊 因为下边有pop 千万不要把原始数据给修改掉了
    corpus = []
    max_turn = 0
    for sample in batch:
        turns = sample['turns']
        if turns >= max_turn:
            max_turn = turns
    for i in range(max_turn):
        utterance_set = []
        length_set = []
        for sample in batch:
            dialogue = sample['dialogue']
            if len(dialogue) >= 1:
                dialogue_sample = dialogue.pop()  # 就在这个地方！！！ pop一边之后就没额
                utterance, length = dialogue_sample['utterance'], dialogue_sample['length']
            else:
                utterance, length = [__eot__, ], 1
            utterance_set.append(torch.tensor(utterance))
            length_set.append(length)
        assert len(utterance_set) == len(batch)
        assert len(length_set) == len(batch)
        corpus.append(
            [pad_sequence(utterance_set, batch_first=True, padding_value=total_words),
             torch.tensor(length_set)])  # 完事之后记得reverse ！！！！！！！！！！！！！！！

    while len(corpus) > max_turns:
        corpus.pop()
    corpus.reverse()
    return corpus


def valid(args, model, device, valid_loader):
    model.eval()
    average_loss = 0
    ppl_loss, N_w = 0, 0
    with torch.no_grad():
        for batch_idx, corpus in enumerate(valid_loader):
            response_y = corpus_shift_right(corpus, args.bs)
            for i, (utterance, length) in enumerate(corpus):
                corpus[i] = (utterance.to(device), length.to(device))  # to(device)不是原地操作
            response_y = response_y.to(device)
            prob = model(corpus)
            loss = F.nll_loss(prob.view(-1, total_words + 1), response_y.view(-1), ignore_index=args.total_words,
                              reduction='mean')
            ppl_loss += F.nll_loss(prob.view(-1, total_words + 1), response_y.view(-1), ignore_index=args.total_words,
                                   reduction='sum').item()
            N_w += corpus[-1][-1].sum().item()
            average_loss += loss.item()
        average_loss /= len(valid_loader)
        ppl = math.exp(ppl_loss / N_w)
        print('Validation Average Loss: {:.6f}\t PPL: {:.6f}'.format(average_loss, ppl))
    return average_loss, ppl


def corpus_shift_right(corpus, batch_size):
    response_corpus = corpus[-1]
    origin_response = response_corpus[0]
    temp_response = origin_response[:, 1:].clone().detach()
    eot_tensor = torch.tensor([__eot__ for _ in range(batch_size)])
    response_y = torch.cat((temp_response, torch.unsqueeze(eot_tensor, 1)),
                           dim=1)  # target to cac loss
    return response_y


def reformat_corpus(context, batch_size, ys, ys_length):
    return context.append([ys, ys_length])


def infer(args, model, device, test_loader, n, dic):
    model.eval()
    generation = []
    with torch.no_grad():
        for batch_idx, corpus in enumerate(test_loader):
            ys = torch.ones(1).fill_(__eot__).type_as(corpus[-1][0])
            ys_length = torch.tensor([len(ys)])
            new_corpus = copy.deepcopy(corpus[:-1])
            new_corpus.append([ys.unsqueeze(0), ys_length])
            for i, (utterance, length) in enumerate(new_corpus):
                new_corpus[i] = (utterance.to(device), length.to(device))
            out = model(new_corpus).squeeze(0)  # l,total_word
            temp_prob = out[-1, :]
            next_word_probs, next_words = torch.topk(temp_prob, n, dim=-1)  # n
            ys_objs = []
            for next_word_prob, next_word in zip(next_word_probs, next_words):
                if next_word == __eot__:
                    ys_objs.append(
                        {'track': torch.cat((ys.to(device), next_word.unsqueeze(0))), 'prob': next_word_prob.item(),
                         'generate': False})  # 这里貌似并不是概率 因为取log了
                else:
                    ys_objs.append({
                        'track': torch.cat((ys.to(device), next_word.unsqueeze(0))),
                        'prob': next_word_prob.item(),
                        'generate': True
                    })
            ys_objs = beam_search(args, corpus, ys_objs, n, model, 2, device)
            ys_objs.sort(key=getprob, reverse=True)
            final_ys = ys_objs[0]
            print('Max Prob = {}'.format(math.exp(final_ys['prob'])))
            track = torch.squeeze(final_ys['track']).tolist()  # track 是 一个生成的数字序列
            print(track)
            print(sentence(dic, track))
            generation.append(track)

        return generation


def beam_search(args, corpus, ys_objs, n, model, l, device):
    if l == max_sentence_length:
        assert len(ys_objs) == n
        return ys_objs
    new_objs = []
    for ys_obj in ys_objs:
        ys = ys_obj['track']
        prob = ys_obj['prob']
        generate = ys_obj['generate']

        if generate is not True:
            new_objs.append({'track': ys, 'prob': prob, 'generate': False})
            continue

        ys_length = torch.tensor([len(ys)])
        new_corpus = copy.deepcopy(corpus[:-1])
        new_corpus.append([ys.unsqueeze(0), ys_length])
        for i, (utterance, length) in enumerate(new_corpus):
            new_corpus[i] = (utterance.to(device), length.to(device))

        out = model(new_corpus).squeeze(0)  # l,total_word
        temp_prob = out[-1, :]

        next_word_probs, next_words = torch.topk(temp_prob, n, dim=-1)  # n

        for next_word_prob, next_word in zip(next_word_probs, next_words):

            if next_word == __eot__:
                new_objs.append(
                    {'track': torch.cat((ys, next_word.unsqueeze(0))), 'prob': prob + next_word_prob.item(),
                     'generate': False})
            else:
                new_objs.append({'track': torch.cat((ys, next_word.unsqueeze(0))), 'prob': prob + next_word_prob.item(),
                                 'generate': True})
    ys_objs = new_objs
    ys_objs.sort(key=getprob, reverse=True)
    ys_objs = ys_objs[:n]
    return beam_search(args, corpus, ys_objs, n, model, l + 1, device)


def getprob(elem):
    return elem['prob']


def sentence(dic, sen):
    temp = []
    for idx in sen:
        temp.append(dic[idx])
    temp = ' '.join(temp)
    return temp
