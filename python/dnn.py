import collections
import numpy as np
import os
import pickle
import random
import scipy.stats
import sqlite3
import time

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

dvc = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('dnn module loaded with this device:', dvc)
crit = nn.SmoothL1Loss(reduction='sum')

import cfg


##################
# NOTES/CONCEPTS #
##################
# 
# 'triplets' of feature vectors:
#     index 0: turn-initial IPU of speaker A
#     index 1: turn-final IPU of next turn by speaker B
#     index 2: turn-initial IPU of next turn by speaker A
# A, B is independent of the labels in Fisher; each speaker takes either 
#     position for some triplets for each conversation
# for some data, index 0 represents the very first IPU ('IPU0') of speaker A
#
# three different subsets of Fisher:
#     'trn': training, ~80%
#     'vld': validation, ~10%
#     'tst': test, ~10%
# no balancing by gender etc., percentage based on session count, not IPU count
#
# different types of networks (used as components of larger networks):
#     'A':  generates prediction 2_A for vector 2 from vector 0
#     'AB': generates prediction 2_AB for vector 2 from 2_A and 2_B
#     'AS': adversarial selector, predicts vectors 0 and 2 from vector 1
#           (with gradient reversal for prediction of vector 1)


################################################################################
#                            AUX CLASS DEFINITIONS                             #
################################################################################

class ChunkTriplets (torch.utils.data.Dataset):
    def __init__(self, data):
        ''' class needed to use dataloader; expects triplets of data '''
        super(ChunkTriplets, self).__init__()
        self.data = data
    
    def __getitem__(self, index):
        return (self.data[index][0], 
                self.data[index][1], 
                self.data[index][2])

    def __len__(self):
        return len(self.data)


class GradientReversal(torch.autograd.Function):
    def __init__(self):
        ''' component to invert sign of gradient '''
        super(GradientReversal, self).__init__()

    @staticmethod
    def forward(self, fea):
        return fea.clone()

    @staticmethod
    def backward(self, grad):
        return -grad.clone()


################################################################################
#                            MAIN CLASS DEFINITIONS                            #
################################################################################


class FlexFFNN(nn.Module):
    def __init__(self, dims, bn_relu_flags, out_i=0, bn_before_relu=True):
        ''' feedforward network of linear and batch norm layers with relu 
            activations; size, number of layers and position of batch
            norm + activation are flexible; allows return of one
            intermediate output in addition to final output
        
        update save function if parameters change!
        
        args:
            dims: list of int; dimensions of the len(dims)-1 linear and 
                up to len(dims)-2 batchnorm layers of the network;
                i-th layer: dims[i] inputs to dims[i+1] outputs
            bn_relu_flags: list of len(dims)-2 booleans; entry i: batch 
                norm and relu after linear layer i, yes or no
            out_i: function returns output of linear layer with this 
                index (0 base) in addition to final output (allows return 
                of intermediate result, i.e., encoding of the input)
            bn_before_relu: batch norm before relu (True) or after (False)
                (there seems to be debate over which should be done first)
        '''

        super(FlexFFNN, self).__init__()
        
        self.dims = dims
        self.bn_relu_flags = bn_relu_flags
        # create layers as dynamic attributes (so they are recognized by
        # the pytorch api) and store them in a list (for convenience)
        self.layers = []
        for i in range(len(dims)-1):
            setattr(self, 'l%d' % i, nn.Linear(dims[i], dims[i+1]))
            self.layers.append(getattr(self, 'l%d' % i))
            if i < len(dims)-2 and bn_relu_flags[i]:
                setattr(self, 'bn%d' % i, nn.BatchNorm1d(dims[i+1]))
                self.layers.append(getattr(self, 'bn%d' % i))
        self.out_i = out_i
        self.bn_before_relu = bn_before_relu
        
        
    def forward(self, fea):
        o = fea
        e = None
        i = 0
        # apply layers, differentiate between linear and batch norm
        for l in self.layers:
            if isinstance(l, nn.Linear):
                o = l(o)
                if i == self.out_i:
                    # store this input encoding to return later
                    e = o
                i += 1
            else:
                # apply relu and batch norm, in selected order
                o = F.relu(l(o)) if self.bn_before_relu else l(F.relu(o))
        return e, o
    
    def save(self, name):
        # write instance config and network state dictionary to files
        kwargs = {
            'dims': self.dims,
            'bn_relu_flags': self.bn_relu_flags,
            'out_i': self.out_i,
            'bn_before_relu': self.bn_before_relu
        }
        with open('%s%s.pickle' % (cfg.NETS_DIR, name), 'wb') as pickle_file:
            pickle.dump(kwargs, pickle_file)
        torch.save(self.state_dict(), '%s%s.pt' % (cfg.NETS_DIR, name))
    
    @staticmethod
    def load(name, dvc):
        # read instance config and network state dictionary from files
        with open('%s%s.pickle' % (cfg.NETS_DIR, name), 'rb') as pickle_file:
            kwargs = pickle.load(pickle_file)
        net = FlexFFNN(**kwargs).to(dvc)
        net.load_state_dict(torch.load('%s%s.pt' % (cfg.NETS_DIR, name)))
        return net


class AdversarialSelectorNet(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2):
        super(AdversarialSelectorNet, self).__init__()
        
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        
        self.l0 = nn.Linear(input_size, hidden_size1)
        self.bn0 = nn.BatchNorm1d(hidden_size1)
        self.l1 = nn.Linear(hidden_size1, hidden_size2)
        self.l2_1 = nn.Linear(hidden_size2, hidden_size1)
        self.bn2_1 = nn.BatchNorm1d(hidden_size1)
        self.l3_1 = nn.Linear(hidden_size1, input_size)
        self.gr = GradientReversal.apply
        self.l2_2 = nn.Linear(hidden_size2, hidden_size1)
        self.bn2_2 = nn.BatchNorm1d(hidden_size1)
        self.l3_2 = nn.Linear(hidden_size1, input_size)


    def forward(self, fea):
        e = self.l1(F.relu(self.bn0(self.l0(fea))))
        o1 = self.l3_1(F.relu(self.bn2_1(self.l2_1(e))))
        o2 = self.l3_2(F.relu(self.bn2_2(self.l2_2(self.gr(e)))))
        
        return o1, o2
    
    def save(self, name):
        # write instance config and network state dictionary to files
        kwargs = {
            'input_size': self.input_size,
            'hidden_size1': self.hidden_size1,
            'hidden_size2': self.hidden_size2
        }
        with open('%s%s.pickle' % (cfg.NETS_DIR, name), 'wb') as pickle_file:
            pickle.dump(kwargs, pickle_file)
        torch.save(self.state_dict(), '%s%s.pt' % (cfg.NETS_DIR, name))
    
    @staticmethod
    def load(name, dvc):
        # read instance config and network state dictionary from files
        with open('%s%s.pickle' % (cfg.NETS_DIR, name), 'rb') as pickle_file:
            kwargs = pickle.load(pickle_file)
        net = AdversarialSelectorNet(**kwargs).to(dvc)
        net.load_state_dict(torch.load('%s%s.pt' % (cfg.NETS_DIR, name)))
        return net


################################################################################
#                           TRAINING AND EVALUATION                            #
################################################################################

def get_data_loader(data_set, subset, shuffle=True):
    ''' creates DataLoader for subset of given data, shuffled if needed '''
    return torch.utils.data.DataLoader(
        ChunkTriplets(data_set[subset]), cfg.BATCH_SIZE, shuffle=shuffle)


def train(nets, opt, crit, loader, epochs, dvc):
    ''' train a network using whole dataset for a certain number of epochs
    
    args:
        nets: dict of networks indexed 'A', 'AB', and 'AS'
            (also see note at the top)
            only one will be trained, training of 'AB' can use 'A';
            missing key in dict => 'empty' network, identity function
            examples:
                to train 'A'
                    pass {'A': <untrained_net_a>}
                to train 'AB', utilizing fully trained net 'A'
                    pass {'A': <trained_net_a>, 'AB': <untrained_net_ab}
                to train 'AB', utilizing no net 'A'
                    pass {'AB': <untrained_net_ab}
        opt: optimizer
        crit: optimization criterion, loss function
        loader: dataloader to yield training examples
        epochs: number of times to run through entire loader dataset
        dvc: device to execute computations
    '''
    losses = []
    for epoch in range(epochs):
        loss_total = 0
        for in_a, in_b, out in loader:
            opt.zero_grad()
            if 'A' in nets:
                # apply net 'A' and compute loss unless training 'AB'
                out_a = nets['A'](in_a.to(dvc))[1]
                if 'AB' not in nets:
                    loss = crit(out_a, out.to(dvc))
            else:
                out_a = in_a.to(dvc)
            out_b = in_b.to(dvc)
            if 'AB' in nets:
                # train net 'AB'; input = outputs of 'A' and 'identity'
                in_ab = torch.cat((out_a, out_b), 1)
                loss = crit(nets['AB'](in_ab)[1], out.to(dvc))
            if 'AS' in nets:
                # train adversarial selector
                o1, o2 = nets['AS'](in_b.to(dvc))
                loss = crit(o1, out.to(dvc)) + crit(o2, in_a.to(dvc))
            loss.backward()
            opt.step()
            loss_total += loss.item()
        losses.append(loss_total / loader.dataset.__len__())
    return losses


def evaluate(nets, crit, loader, dvc):
    ''' evaluate a network by computing losses for given dataset
    
    was used to compare different depths/widths, not used anymore
    compare with 'train' function for arguments/concepts;
    '''
    # store current mode, then put networks in eval mode
    modes = {}
    for key, net in nets.items():
        modes[key] = net.training
        net.eval()
    
    losses = {'A': [], 'AB': [], 'AS': []}
    with torch.set_grad_enabled(False):
        for in_a, in_b, out in loader:
            if 'A' in nets:
                out_a = nets['A'](in_a.to(dvc))[1]
                losses['A'].append(crit(out_a, out.to(dvc)).item())
            else:
                out_a = in_a.to(dvc)
            out_b = in_b.to(dvc)
            if 'AB' in nets:
                losses['AB'].append(
                    crit(nets['AB'](torch.cat((out_a, out_b), 1))[1],
                         out.to(dvc)).item())
            if 'AS' in nets:
                o1, o2 = nets['AS'](in_b.to(dvc))
                losses['AS'].append(
                    crit(o1, out.to(dvc)) + crit(o2, in_a.to(dvc)))
    length = loader.dataset.__len__()
    losses = {
        'A': sum(losses['A']) / length if 'A' in nets else None, 
        'AB': sum(losses['AB']) / length if 'AB' in nets else None, 
        'AS': sum(losses['AS']) / length if 'AS' in nets else None
    }
    # put networks back in training mode if they were before
    for key, net in nets.items():
        if modes[key]:
            net.train()
    return losses


def shuffle(data_real):
    ''' shuffles middle ipu in triplets per session to create fake sessions '''
    data_fake = collections.defaultdict(list)
    for ses_id, speaker_triplets in data_real.items():
        for a_or_b in ['A', 'B']:
            # extract data for current speaker
            rows_real = speaker_triplets[a_or_b]
            # extract three columns from all rows/triplets of features
            cols_real = list(zip(*rows_real))
            # shuffle middle column
            cols_fake = [random.sample(col, len(col)) if i == 1 else col
                         for i, col in enumerate(cols_real)]
            # recombine columns to triplets, add to results
            data_fake[ses_id] += list(zip(*cols_fake))
    return data_fake


def add_losses(losses, triplets, nets, crit):
    ''' adds losses for given triplets to intermeditate result '''
    loader = torch.utils.data.DataLoader(
        ChunkTriplets(triplets), cfg.BATCH_SIZE, shuffle=False)
    for in_a, in_b, out in loader:
        # residualization approach
        out_a = nets['A'](in_a.to(dvc))[1].detach()
        tmp1 = crit(out_a, out.to(dvc)).item()
        in_ab = torch.cat((out_a, in_b.to(dvc)), 1)
        out_ab = nets['AB'](in_ab.to(dvc))[1].detach()
        tmp2 = crit(out_ab, out.to(dvc)).item()
        losses[cfg.MEASURE_ID_DR] += tmp2 - tmp1
        # adversarial selector approach
        o1, o2 = nets['AS'](in_b.to(dvc))
        tmp3 = crit(o1, out.to(dvc)).item()
        tmp4 = crit(o2, in_a.to(dvc)).item()
        losses[cfg.MEASURE_ID_AS] += tmp3 - tmp4


def run_fake_ses_test(loader_trn, data_set_ses_fc, data_ses_gc, runs, epochs):
    ''' runs fake ses. detection test for both measures on fisher and games '''
    # load or init results log
    if os.path.isfile(cfg.SES_TEST_RES_FNAME):
        with open(cfg.SES_TEST_RES_FNAME, 'rb') as res_file:
            res = pickle.load(res_file)
        print('loaded results file with results for %d out of %d runs' %
              (len(res[cfg.CORPUS_ID_FISHER][cfg.MEASURE_ID_DR]), runs))
    else:
        print('no results file found, starting from scratch')
        res = {c : {m: [] for m in cfg.MEASURE_IDS} for c in cfg.CORPUS_IDS}

    print('starting', time.ctime())
    for run in range(len(res[cfg.CORPUS_ID_FISHER][cfg.MEASURE_ID_DR]), runs):
        # train the networks needed for the two measures
        net_a = train_net('A', loader_trn, epochs, run)
        net_ab = train_net('AB', loader_trn, epochs, run)
        net_as = train_net('AS', loader_trn, epochs, run)
        nets = {
            'A': net_a.eval(),
            'AB': net_ab.eval(),
            'AS': net_as.eval()
        }
        
        # run the test on the fisher corpus test set and the games corpus
        configs = [
            (cfg.CORPUS_ID_FISHER, data_set_ses_fc['tst']),
            (cfg.CORPUS_ID_GAMES, data_ses_gc)
        ]
        for corpus_id, data_real in configs:
            # create fake sessions and init log of correctly identified sessions
            data_fake = shuffle(data_real)
            correct_cnt = {m: 0 for m in cfg.MEASURE_IDS}
            for ses_id, speaker_triplets in data_real.items():
                # compute both measures for both real and fake sessions
                losses_real = {m: 0 for m in cfg.MEASURE_IDS}
                add_losses(losses_real, speaker_triplets['A'], nets, crit)
                add_losses(losses_real, speaker_triplets['B'], nets, crit)
                losses_fake = {m: 0 for m in cfg.MEASURE_IDS}
                add_losses(losses_fake, data_fake[ses_id], nets, crit)
                # determine whether measures correctly identified fake sessions
                for m in cfg.MEASURE_IDS:
                    if losses_real[m] < losses_fake[m]:
                        correct_cnt[m] += 1
            # log fraction of correctly identified sessions for this run 
            for m in cfg.MEASURE_IDS:
                res[corpus_id][m] += [correct_cnt[m] / len(data_fake)]
        # print and write status updates
        print(run+1, 'done', time.ctime())
        with open(cfg.SES_TEST_RES_FNAME, 'wb') as pickle_file:
            pickle.dump(res, pickle_file)
        with open('log.txt', 'w') as log_file:
            for m in cfg.MEASURE_IDS:
                for c in cfg.CORPUS_IDS:
                    log_file.write('%s %s %.4f (%.4f)\n' % 
                                   (m, c, 
                                    np.mean(res[c][m]), np.std(res[c][m])))
    print('done', time.ctime())


def run_corr_test(loader_trn, data_tsk_gc, runs, epochs):
    ''' runs correlation test for both measures '''
    # load or init results log
    if os.path.isfile(cfg.COR_TEST_RES_FNAME):
        with open(cfg.COR_TEST_RES_FNAME, 'rb') as res_file:
            res = pickle.load(res_file)
        print('loaded res file with results for %d out of %d runs' %
              (len(res), runs))
    else:
        print('no res file found, starting from scratch')
        res = []

    # sql to get two result columns per social variable, one for each speaker
    # ('a_or_b' marks describer; in tasks where A is describer, get describer 
    #  annotations for A and follower annotations for B; and vice versa)
    sql_fragment1 = ', '.join(['CASE ' \
                               '    WHEN a_or_b == "A" ' \
                               '    THEN %s_describer_yes ' \
                               '    ELSE %s_follower_yes ' \
                               'END %s_A, ' \
                               'CASE ' \
                               '    WHEN a_or_b == "B" ' \
                               '    THEN %s_describer_yes ' \
                               '    ELSE %s_follower_yes ' \
                               'END %s_B' % 
                               (v[1], v[1], v[0], v[1], v[1], v[0])
                               for v in cfg.SOCIAL_VARS])
    # load all tasks with annotations in dict, one list per tsk_id;
    # leave first four entries empty for both measures for both speakers
    with sqlite3.connect(cfg.DB_FNAME_GC) as conn:
        c = conn.cursor()
        c.execute('SELECT tsk_id, ' + sql_fragment1 + ' FROM tasks;')
        tasks = {row[0]: 4*[None] + [v for v in row[1:]] 
                 for row in c.fetchall()}

    print('starting', time.ctime())
    for run in range(len(res), runs):
        # train the networks needed for the two measures
        # net_a = train_net('A', loader_trn, epochs)
        # net_ab = train_net('AB', loader_trn, epochs)
        # net_as = train_net('AS', loader_trn, epochs)
        # nets = {
        #     'A': net_a.eval(),
        #     'AB': net_ab.eval(),
        #     'AS': net_as.eval()
        # }
        nets = {
            'A': FlexFFNN.load('net_a_run_%d' % run, dvc).eval(),
            'AB': FlexFFNN.load('net_ab_run_%d' % run, dvc).eval(),
            'AS': AdversarialSelectorNet.load('net_as_run_%d' % run, dvc).eval()
        }
        # run the measures on all tasks and speakers
        # (cases of missing data are skipped here and below because a few
        #  speakers did not speak in a few tasks)
        for tsk_id in data_tsk_gc:
            for a_or_b in ['A', 'B']:
                if len(data_tsk_gc[tsk_id][a_or_b]) > 0:
                    losses = {m: 0 for m in cfg.MEASURE_IDS}
                    add_losses(losses, data_tsk_gc[tsk_id][a_or_b], nets, crit)
                    triplet_cnt = len(data_tsk_gc[tsk_id][a_or_b])
                    tasks[tsk_id][0 if a_or_b == 'A' else 1] = \
                        losses[cfg.MEASURE_ID_DR] / triplet_cnt
                    tasks[tsk_id][2 if a_or_b == 'A' else 3] = \
                        losses[cfg.MEASURE_ID_AS] / triplet_cnt
        # compute correlations for both measures and all social variables
        sigs = {m: [] for m in cfg.MEASURE_IDS}
        full = {m: [] for m in cfg.MEASURE_IDS}
        for m in cfg.MEASURE_IDS:
            for i, var in enumerate(cfg.SOCIAL_VARS):
                iA = 0 if m == cfg.MEASURE_ID_DR else 2
                iB = 1 if m == cfg.MEASURE_ID_DR else 3
                x = [val[iA] for val in tasks.values() if val[iA]] + \
                    [val[iB] for val in tasks.values() if val[iB]]
                y = [val[4+2*i] for val in tasks.values() if val[iA]] + \
                    [val[5+2*i] for val in tasks.values() if val[iB]]
                r, p = scipy.stats.pearsonr(x, y)
                if p < 0.05:
                    sigs[m] += [(var, r, p)]
                full[m] += [(var, r, p)]
            # print a status update with correlations with p < 0.05
            print(run, m, sigs[m], time.ctime())
        # write full results to file and print status update
        res += [{m: full[m] for m in cfg.MEASURE_IDS}]
        with open(cfg.COR_TEST_RES_FNAME, 'wb') as res_file:
            pickle.dump(res, res_file)
        print(run+1, 'done', time.ctime())
    print('done', time.ctime())


def print_corr_test_results():
    ''' prints summary of correlation test results '''
    # load results file
    with open(cfg.COR_TEST_RES_FNAME, 'rb') as res_file:
        res = pickle.load(res_file)
    for m in cfg.MEASURE_IDS:
        print(m)
        # compute adjusted p values per test for each run in the results
        for i in range(len(res)):
            # sort results of given run by p values
            entry = sorted(res[i][m], key=lambda x: x[2])
            for var in cfg.SOCIAL_VARS:
                # add adjusted values to results, in order of SOCIAL_VARS list
                alpha = [entry[j][2] * len(entry) / (j+1)
                         for j in range(len(entry)) 
                         if entry[j][0][0] == var[0]]
                res[i][m] += alpha
        # determine significant results per social variable  
        for i, var in enumerate(cfg.SOCIAL_VARS):
            # sort results by adjusted p values for current social variable
            res_srt = sorted(res, key=lambda x: x[m][3+i])
            # adjust p values again based on number of overall runs
            alphas = [res_srt[j][m][3+i] * len(res) / (j+1)
                      for j in range(len(res))]
            # determine number of significant results and number per valence
            k = max([j if alphas[j] < 0.05 else -1 
                     for j in range(len(alphas))])
            pos_entries = [(j, res_srt[j][m][i]) for j in range(k) 
                           if res_srt[j][m][i][1] > 0]
            neg_entries = [(j, res_srt[j][m][i]) for j in range(k) 
                           if res_srt[j][m][i][1] < 0]
            # print overview of significant results
            print('\t%s sig. %d times, %d pos., %d neg' %
                  (var[0], 0 if k == -1 else k, 
                   len(pos_entries), len(neg_entries)))
            if len(pos_entries) > 0:
                print('\tmost sig. pos. entry at index %d: %s' %
                      (pos_entries[0][0], pos_entries[0][1]))
            if len(neg_entries) > 0:
                print('\tmost sig. neg. entry at index %d: %s' %
                      (neg_entries[0][0], neg_entries[0][1]))
            # print extrema of raw p values
            res_srt = sorted(res, key=lambda x: x[m][i][2])
            print('\tentry with lowest p for %s: %s\n' %
                  (var[0], res_srt[0][m]))
            print('\tentry with highest p for %s: %s\n' %
                  (var[0], res_srt[-1][m]))
        print('\n')


def train_net(net_type, loader_trn, epochs, run):
    ''' trains network of requested type with given data for given no. of epochs

    net_type 'AB' requires that net called 'net_a' of type 'A' can be loaded'''

    if net_type == 'AS':
        net = AdversarialSelectorNet(
            cfg.IN_SIZE, cfg.HIDDEN_SIZE1, cfg.HIDDEN_SIZE2).to(dvc)
    else:
        # 'A' or 'AB' net requested
        # 'AB' uses double input size, all other parameters are the same
        dims = [
            (2*cfg.IN_SIZE) if net_type == 'AB' else cfg.IN_SIZE, 
            cfg.HIDDEN_SIZE1, 
            cfg.HIDDEN_SIZE2, 
            cfg.HIDDEN_SIZE1, 
            cfg.IN_SIZE
        ]
        net = FlexFFNN(dims, [True, False, True], 0, True).to(dvc) 
    nets = {net_type: net}
    if net_type == 'AB':
        # nets of type 'AB' are trained with pretrained 'A' component
        net_a = FlexFFNN.load('net_a_run_%d' % run, dvc).eval()
        for param in net_a.parameters():
            param.requires_grad = False
        nets['A'] = net_a
    opt = optim.Adam(net.parameters())
    losses = train(nets, opt, crit, loader_trn, epochs, dvc)
    net.save('net_%s_run_%d' % (net_type.lower(), run))
    return net




