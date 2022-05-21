#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/8/10 14:27
# @Author  : name
# @File    : run_bpr.py
import argparse
import collections

import pandas as pd
import torch

from torch.utils.data import DataLoader
from Data import Kwai,Tmall, load_user_item_list,TmallSlot
from models.MF import MF_multi, MF, PDA
from models.MACR import MACRMF
from models.AdFair import AdFair, Discriminator
from torch.optim import Adam
from torch.nn import MSELoss, BCELoss
from sklearn import metrics
import numpy as np
import os
import random as rd
from tqdm import trange
from used_metric import get_performance
import matplotlib.pyplot as plt 

SEED=2
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark = False

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
path_data = '../tmall/tmall_16_time'
# path_model = 'save_tmall/16_MACRESMM0.1.pt'
# path_model = 'save_tmall/16_ADFair_ESMMstep1-1.pt'
# path_model = 'save_tmall/PDA2.pt'
path_model = 'save_tmall/MBD_doI_0.3.pt'

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument("--test_user_item_list", type=str, default=os.path.join(path_data, 'test.txt'))
parser.add_argument("--train_user_item_list", type=str, default=os.path.join(path_data,'train.txt'))
parser.add_argument("--valid_user_item_list", type=str, default=os.path.join(path_data,'valid.txt'))
parser.add_argument("--test_user_item_buy", type=str, default=os.path.join(path_data, 'test_buy.txt'))
parser.add_argument("--train_user_item_buy", type=str, default=os.path.join(path_data,'train_buy.txt'))
parser.add_argument("--valid_user_item_buy", type=str, default=os.path.join(path_data, 'valid_buy.txt'))
parser.add_argument("--valid_batch_num", type=str, default=10)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--embedding_dim", type=int, default=128)
parser.add_argument("--batch_size", type=int, default=1024 * 4)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--decay", type=float, default = 1 * 1e-6) # 4
# parser.add_argument("--model", type=str, default='PDA')
parser.add_argument("--model", type=str, default='MF_multi')
parser.add_argument("--eval", type=str, default='buy')
# parser.add_argument("--loss", type=str, default='IPS+DR')
parser.add_argument("--loss", type=str, default='multi')
parser.add_argument("--patience", type=int, default=10)
parser.add_argument("--gamma", type=float, default=0.4)
# parser.add_argument("--gamma", type=float, default=0.1)
# parser.add_argument("--gammal", type=float, default=1.2)
parser.add_argument("--gammal", type=float, default=1.2)
parser.add_argument("--gammac", type=float, default=0.5)
parser.add_argument("--withratio", type=bool, default=True)
parser.add_argument("--poplike", type=bool, default=True)
parser.add_argument('--ablation', type=bool, default=True)
parser.add_argument("--ratiodo", type=float, default=0.3)
## MACR
parser.add_argument("--c", type=float, default=0.1)
parser.add_argument("--alpha", type=float, default=1)
parser.add_argument("--beta", type=float, default=1)
## AdFair
parser.add_argument("--lamb", type=float, default=1)
parser.add_argument("--steps", type=int, default=1)
args = parser.parse_args()

keys = ['u_id', 'i_id', 'time', 'buy', 'click']
# keys = ['u_id', 'i_id', 'time', 'buy', 'click', 'pop_tmp', 'pop_pre']

data_train = pd.read_csv(os.path.join(path_data,'train_interaction_filter.txt'), sep=' ')
data_valid = pd.read_csv(os.path.join(path_data,'valid_interaction_filter.txt'), sep=' ')
data_test = pd.read_csv(os.path.join(path_data,'test_interaction_filter.txt'), sep=' ')

# nomalization of ratio/click/post-click(buy)
item_ratio = pd.read_csv(os.path.join(path_data, 'tmall_itemid.csv'), sep=' ')
item_pop_time = pd.read_csv(os.path.join(path_data, 'kwai_item_id_pop2.csv'), sep='\t')
item_ratio = pd.read_csv(os.path.join(path_data, 'tmall_itemid_ratio_buys.csv'), sep=' ')

item = list(item_ratio['itemid'].values)
ratio = list(item_ratio['ratio(buy/click)'].values)
ratio = (ratio - np.min(ratio))/(np.max(ratio) - np.min(ratio))

click = list(item_ratio['click'].values)
click = (click - np.min(click))/(np.max(click) - np.min(click))
buy = list(item_ratio['buy'].values)
buy = (buy - np.min(buy))/(np.max(buy) - np.min(buy))
# d_ratio = dict(zip(item, ratio))
ratiopop = list(item_ratio['ratio-pop'].values)

data_train.columns = keys
data_valid.columns = keys
data_test.columns = keys

print('train_num:',len(data_train), 'valid_num:',len(data_valid), 'test_num:', len(data_test))

userNum = len(data_train['u_id'].unique())
itemNum = len(data_train['i_id'].unique())

print('user_num:', userNum, 'item_num:', itemNum)

dataset_train = Tmall(data_train)
dataset_valid = Tmall(data_valid)

# dataset_train = TmallSlot(data_train)
# dataset_valid = TmallSlot(data_valid)

# user clicked items in train/valid/test
eval_click_train = load_user_item_list(args.train_user_item_list)
eval_click_test = load_user_item_list(args.test_user_item_list)
eval_click_valid = load_user_item_list(args.valid_user_item_list)

# users in train/valid/test
users_click_train = list(eval_click_train.keys())
users_click_test = list(eval_click_test.keys())
users_click_valid = list(eval_click_valid.keys())

# user nums in train/valid/test
users_num_train = len(users_click_train)
users_num_test = len(users_click_test)
users_num_valid = len(users_click_valid)
print('click:', 'train user:', users_num_train, 'valid user:', users_num_valid, 'test user:', users_num_test)

eval_buy_train = load_user_item_list(args.train_user_item_buy)
eval_buy_test = load_user_item_list(args.test_user_item_buy)
eval_buy_valid = load_user_item_list(args.valid_user_item_buy)
users_train_buy = list(eval_buy_train.keys())
users_test_buy = list(eval_buy_test.keys())
users_valid_buy = list(eval_buy_valid.keys())
users_num_train_buy = len(users_train_buy)
users_num_test_buy = len(users_test_buy)
users_num_valid_buy = len(users_valid_buy)
print('buy:', 'train buy user:', users_num_train_buy, 'valid buy user:', users_num_valid_buy, 'test buy user:', users_num_test_buy)

def valid_batch(batch_user, batch_result, type='valid', Ks=[20, 50]):
        '''
        batch_rec 的结果是每个用户推荐的topk的结果, 可以通过model.do_recommendation获得
        '''
        result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks)),
                'hit_ratio': np.zeros(len(Ks))}
        for user in range(len(batch_user)):
            u = batch_user[user]
            r = batch_result[user]
            if type == 'train' or type == 'train_buy':
                pass
            else:
                train_item = eval_click_train[u]
                # print('result shape', len(r), 'train_item shape', len(train_item))
                # 踢除 train中的click或buy 的item
                r = r[~np.isin(r, train_item)]
                # print('shape filter:', len(r))

            try:
                if type == 'train':
                    u_target = eval_click_train[u]
                elif type == 'test':
                    u_target = eval_click_test[u]
                elif type == 'valid':
                    u_target = eval_click_valid[u]
                elif type == 'train_buy':
                    u_target = eval_buy_train[u]
                elif type == 'test_buy':
                    u_target = eval_buy_test[u]
                elif type == 'valid_buy':
                    u_target = eval_buy_valid[u]
            except:
                u_target = []
            one_user_result = get_performance(u_target, r, Ks)

            result['precision'] += one_user_result['precision']
            result['recall'] += one_user_result['recall']
            result['ndcg'] += one_user_result['ndcg']
            result['hit_ratio'] += one_user_result['hit_ratio']
        return result

def TopPerformance(tp, model, gamma, withratio=True):
    t = args.valid_batch_num
    # Ks = [20, 50] if tp=='train' or tp == 'test' or tp == 'valid' else [20, 50]
    Ks = [50,100]
    # print(Ks)
    if tp == 'train':
        num_t = int(users_num_train/t)
        users = users_click_train
    elif tp == 'test':
        num_t = int(users_num_test/t)
        users = users_click_test
    elif tp == 'valid':
        num_t = int(users_num_valid/t)
        users = users_click_valid
    elif tp == 'train_buy':
        num_t = int(users_num_train_buy/t)
        users = users_train_buy
    elif tp == 'test_buy':
        num_t = int(users_num_test_buy/t)
        users = users_test_buy
    elif tp == 'valid_buy':
        num_t = int(users_num_valid_buy/t)
        users = users_valid_buy

    users_num = len(users)
    groups = users_num//num_t
    result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks)),
                        'hit_ratio': np.zeros(len(Ks))}
    for i in trange(groups + 1):
        if i == groups:
            batch_users = users[i*num_t:]
            print(len(batch_users))
        else:
            batch_users = users[i*num_t:(i+1)*num_t]
        
        if args.model == 'MF_multi':
            # batch_result = model.do_recommendation(torch.LongTensor(batch_users).cuda(),
            #                                     torch.LongTensor(range(itemNum)).cuda(),
            #                                     1000, args.loss, torch.Tensor(ratio).cuda() ** gamma, withratio)
            if args.ablation == False:
                batch_result = model.do_recommendation(torch.LongTensor(batch_users).cuda(),
                                                    torch.LongTensor(range(itemNum)).cuda(),
                                                    1000, args.loss, torch.Tensor(ratio).cuda() ** gamma, withratio)
            # ablation
            else:
                batch_result = model.do_recommendation_abl(torch.LongTensor(batch_users).cuda(),
                                                    torch.LongTensor(range(itemNum)).cuda(),
                                                    1000, args.loss, torch.Tensor(ratio).cuda() ** gamma,
                                                    # torch.FloatTensor(buy).cuda() ** args.gammal,
                                                    torch.FloatTensor(ratiopop).cuda() ** args.ratiodo,
                                                    withratio)
        if args.model == 'MF':
            batch_result = model.do_recommendation(torch.LongTensor(batch_users).cuda(),
                                                    torch.LongTensor(range(itemNum)).cuda(),
                                                    1000, torch.Tensor(ratio).cuda(), gamma, withratio)
        if args.model == 'MACRMF':
            batch_result = model.do_recommendation(torch.LongTensor(batch_users).cuda(),
                                                    torch.LongTensor(range(itemNum)).cuda(),
                                                    1000, args.c)
                                
        if args.model == 'AdFair':
            batch_result = model.do_recommendation(torch.LongTensor(batch_users).cuda(),
                                                    torch.LongTensor(range(itemNum)).cuda(),
                                                    1000)

        res = valid_batch(batch_users, batch_result.cpu().numpy(), tp, Ks)
        result['precision'] += res['precision']
        result['recall'] += res['recall']
        result['ndcg'] += res['ndcg']
        result['hit_ratio'] += res['hit_ratio']
    result['precision'] = np.multiply(result['precision'], 1 / users_num)
    result['recall'] = np.multiply(result['recall'], 1 / users_num)
    result['ndcg'] = np.multiply(result['ndcg'], 1 / users_num)
    result['hit_ratio'] = np.multiply(result['hit_ratio'], 1 / users_num)
    return result

def performance_rd(tp):
    '''
    random selected item as topk recommended results
    '''
    t = args.valid_batch_num
    Ks = [50,100]
    if tp == 'test_buy':
        num_t = int(users_num_test_buy/t)
        users = users_test_buy
    elif tp == 'valid_buy':
        num_t = int(users_num_valid_buy/t)
        users = users_valid_buy

    users_num = len(users)
    result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks)),
                        'hit_ratio': np.zeros(len(Ks))}

    results = np.random.randint(0, itemNum, size=(len(users), 1000))
    # print(results.shape)
    res = valid_batch(users, results, tp, Ks)
    result['precision'] += res['precision']
    result['recall'] += res['recall']
    result['ndcg'] += res['ndcg']
    result['hit_ratio'] += res['hit_ratio']

    result['precision'] = np.multiply(result['precision'], 1 / users_num)
    result['recall'] = np.multiply(result['recall'], 1 / users_num)
    result['ndcg'] = np.multiply(result['ndcg'], 1 / users_num)
    result['hit_ratio'] = np.multiply(result['hit_ratio'], 1 / users_num)
    return result

def TopPerformance_pd(tp, model, gamma, stage, PDA=True):
    '''
    PDA 模型的评价函数
    '''
    t = args.valid_batch_num
    # Ks = [20, 50] if tp=='train' or tp == 'test' or tp == 'valid' else [20, 50]
    Ks = [50, 100]
    # print(Ks)
    if tp == 'train':
        num_t = int(users_num_train/t)
        users = users_click_train
    elif tp == 'test':
        num_t = int(users_num_test/t)
        users = users_click_test
    elif tp == 'valid':
        num_t = int(users_num_valid/t)
        users = users_click_valid
    elif tp == 'train_buy':
        num_t = int(users_num_train_buy/t)
        users = users_train_buy
    elif tp == 'test_buy':
        num_t = int(users_num_test_buy/t)
        users = users_test_buy
    elif tp == 'valid_buy':
        num_t = int(users_num_valid_buy/t)
        users = users_valid_buy

    users_num = len(users)
    result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks)),
                        'hit_ratio': np.zeros(len(Ks))}
    for i in trange(t):
        batch_users = users[i*num_t:(i+1)*num_t]        
        if stage == 'valid':
            pop = list(item_pop_time['pop_test'].values)
        elif stage == 'test':
            pop = list(item_pop_time['pop_test'].values)
        else:
            pop = list(item_pop_time['pop_train'].values)
        # pop = buy
        batch_result = model.do_recommendation(
                torch.LongTensor(batch_users).cuda(),torch.LongTensor(range(itemNum)).cuda(),
                1000, torch.Tensor(pop).cuda(), gamma,
            )

        res = valid_batch(batch_users, batch_result.cpu().numpy(), tp, Ks)
        result['precision'] += res['precision']
        result['recall'] += res['recall']
        result['ndcg'] += res['ndcg']
        result['hit_ratio'] += res['hit_ratio']
    result['precision'] = np.multiply(result['precision'], 1 / users_num)
    result['recall'] = np.multiply(result['recall'], 1 / users_num)
    result['ndcg'] = np.multiply(result['ndcg'], 1 / users_num)
    result['hit_ratio'] = np.multiply(result['hit_ratio'], 1 / users_num)
    return result

def build_optimizer(model,optimizer_name, lr, l2_weight):
    if optimizer_name == 'gd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=l2_weight)
    elif optimizer_name == 'adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=lr, weight_decay=l2_weight)
    elif optimizer_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_weight)
    else:
        assert self.optimizer_name in ['GD', 'Adagrad', 'Adam']
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=l2_weight)
    return optimizer

def train():
    # initial model
    if args.model == 'MF':
        model = MF(userNum, itemNum, args.embedding_dim).cuda()
        optim = Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)

    if args.model == 'MF_multi':
        model = MF_multi(userNum, itemNum, args.embedding_dim, args.loss).cuda()
        optim = Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)

    if args.model == 'MACRMF':
        model = MACRMF(userNum, itemNum, args.embedding_dim).cuda()
        optim = Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    
    if args.model == 'PDA':
        model = PDA(userNum, itemNum, args.embedding_dim).cuda()
        optim = Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    
    if args.model == 'AdFair':
        model = AdFair(userNum, itemNum, args.embedding_dim, 1, args.lamb).cuda()
        discriminator = Discriminator(args.embedding_dim).cuda()

        model.optimizer = build_optimizer(model, 'adam', args.lr, args.decay)
        discriminator.optimizer = build_optimizer(discriminator, 'adam', args.lr, args.decay)

    lossfn = BCELoss()

    # initialize EarlyStopping
    best = 0
    patience = args.patience
    count = 0

    items = range(itemNum)

    res_click = []
    res_buy = []

    # train
    for epoch in range(args.epochs):
        dl = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, pin_memory=True)
        valid_dl = DataLoader(dataset_valid, batch_size= len(data_valid), shuffle=True)

        if count >= patience:
            break

        for id, batch in enumerate(dl):
            model.train()
            # model MF
            if args.model == 'MF':
                optim.zero_grad()
                batch_neg = []
                for u in batch[0]:
                    u_clicked_items = eval_click_train[u]
                    while True:
                        neg_item = rd.choice(items)
                        if neg_item not in u_clicked_items:
                            batch_neg.append(neg_item)
                            break
                if args.eval == 'click':
                    loss = model.create_bpr_loss(
                        batch[0].cuda(), batch[1].cuda(), torch.LongTensor(batch_neg).cuda(),
                        torch.FloatTensor(click[batch[1].numpy()]).cuda() ** args.gammac,
                        torch.FloatTensor(click[batch_neg]).cuda() ** args.gammac,
                        popclick=True,
                    )

                elif args.eval == 'buy':
                    loss = model.create_bce_loss_buy(batch[0].cuda(), batch[1].cuda(), torch.LongTensor(batch_neg).cuda(), 
                                                            batch[2].float().cuda(), 
                                                            torch.FloatTensor(ratio[batch[1].numpy()]).cuda() ** args.gamma,
                                                            torch.FloatTensor(buy[batch[1].numpy()]).cuda() ** args.gammal,
                                                            withratio=args.withratio, popbuy=False,
                                                            )
                loss.backward()
                optim.step()
            # model multi_MF
            if args.model == 'MF_multi':
                optim.zero_grad()
                batch_neg = []
                for u in batch[0]:
                    u_clicked_items = eval_click_train[u]
                    while True:
                        neg_item = rd.choice(items)
                        if neg_item not in u_clicked_items:
                            batch_neg.append(neg_item)
                            break
                if args.loss == 'multi':    # ESMM model
                    loss = model.create_esmm_loss(
                        batch[0].cuda(), batch[1].cuda(), torch.LongTensor(batch_neg).cuda(), batch[2].float().cuda(),
                        # ratio_click,
                        torch.FloatTensor(ratio[batch[1].numpy()]).cuda() ** args.gamma,
                        torch.FloatTensor(buy[batch[1].numpy()]).cuda() ** args.gammal,
                        torch.FloatTensor(click[batch[1].numpy()]).cuda() ** args.gammac,
                        torch.FloatTensor(click[batch_neg]).cuda() ** args.gammac,
                        withratio = args.withratio,
                        poplike=True,
                        popclick=False,
                                                )
                if args.loss == "IPS+DR":
                    loss = model.create_imp_loss(batch[0].cuda(), batch[1].cuda(), torch.LongTensor(batch_neg).cuda(), batch[2].float().cuda())

                loss.backward()
                optim.step()

            if args.model == 'MACRMF':
                batch_neg = []
                for u in batch[0]:
                    u_clicked_items = eval_click_train[u]
                    while True:
                        neg_item = rd.choice(items)
                        if neg_item not in u_clicked_items:
                            batch_neg.append(neg_item)
                            break
                optim.zero_grad()
                loss = model.create_bce_loss_buy(batch[0].cuda(), batch[1].cuda(), torch.LongTensor(batch_neg).cuda(), batch[2].float().cuda(), args.alpha, args.beta)
                loss.backward()
                optim.step()
            
            if args.model == 'AdFair':
                batch_neg = []
                for u in batch[0]:
                    u_clicked_items = eval_click_train[u]
                    while True:
                        neg_item = rd.choice(items)
                        if neg_item not in u_clicked_items:
                            batch_neg.append(neg_item)
                            break

                discriminator.train()
                model.optimizer.zero_grad()
                itemIdx = batch[1].cuda()
                # rec_loss = model(batch[0].cuda(), itemIdx, batch[2].float().cuda())
                rec_loss = model.create_loss(batch[0].cuda(), batch[1].cuda(), torch.LongTensor(batch_neg).cuda(), batch[2].float().cuda())
                # mask = torch.FloatTensor([[1]] * len(itemIdx)).cuda()
                # iembed = model.apply_filter(model.iEmbed(itemIdx), mask)
                iembed =  model.iEmbed(itemIdx)
                pelnaty = discriminator(iembed, torch.FloatTensor(buy[batch[1].numpy()]).cuda())
                # loss = rec_loss 
                loss = rec_loss - args.lamb * pelnaty
                loss.backward()
                model.optimizer.step()

                for _ in range(args.steps):
                    discriminator.optimizer.zero_grad()
                    disc_loss = discriminator(iembed.detach(), torch.FloatTensor(buy[batch[1].numpy()]).cuda())
                    disc_loss.backward(retain_graph=False)
                    discriminator.optimizer.step()
            
            if args.model == 'PDA':
                optim.zero_grad()
                batch_neg = []
                for u in batch[0]:
                    # 针对 click 的 负例
                    u_clicked_items = eval_click_train[u]
                    while True:
                        neg_item = rd.choice(items)
                        if neg_item not in u_clicked_items:
                            batch_neg.append(neg_item)
                            break
                
                loss = model.bpr_loss(
                    batch[0].cuda(), batch[1].cuda(), torch.LongTensor(batch_neg).cuda(),
                    batch[4].cuda() ** args.gammal,
                    # torch.FloatTensor(buy[batch[1].numpy()]).cuda() ** args.gammal,
                    batch[2].double().cuda(),
                    poplike = args.poplike
                )
                loss.backward()
                optim.step()

            model.eval()
            if id % 200 == 0 and id > 0:
                # print('epoch:', epoch, ' batch:', id, 'Loss: %.4f' % loss.item(), 'penalty: %.4f' % pelnaty.item())   #AdFair
                print('epoch:', epoch, ' batch:', id, 'Loss: %.4f' % loss.item())

            if id % 200 == 0 and id > 0:
                with torch.no_grad():
                    if args.model != 'PDA':
                        result_train = TopPerformance('train', model, args.gamma, args.withratio)
                        result_valid = TopPerformance('valid', model, args.gamma, args.withratio)
                        result_test = TopPerformance('test', model, args.gamma, args.withratio)
                        
                        result_buy_train = TopPerformance('train_buy', model, args.gamma, args.withratio)
                        result_buy_valid = TopPerformance('valid_buy', model, args.gamma, args.withratio)
                        result_buy_test = TopPerformance('test_buy', model, args.gamma, args.withratio)

                    else:
                        # result_train = TopPerformance_pd('train', model, args.gamma, 'train', True)
                        # result_valid = TopPerformance_pd('valid', model, args.gamma, 'valid', True)
                        # result_test = TopPerformance_pd('test', model, args.gamma, 'test', True)
                        
                        result_buy_train = TopPerformance_pd('train_buy', model, args.gamma, 'train', True)
                        result_buy_valid = TopPerformance_pd('valid_buy', model, args.gamma, 'valid', True)
                        result_buy_test = TopPerformance_pd('test_buy', model, args.gamma, 'test', True)
                    
                    # print('click_train:', '\n', result_train)
                    # print('click valid:', '\n', result_valid)
                    # print('click test:', '\n', result_test)

                    print('buy train:', '\n', result_buy_train)
                    print('buy valid:', '\n', result_buy_valid)
                    print('buy test:', '\n', result_buy_test)

                    # torch.save(model, path_model)
                    # val_click = result_valid['precision'][1] + result_valid['recall'][1]
                    val = result_buy_valid['precision'][1] + result_buy_valid['recall'][1]
                    
                    # res_click.append(val_click)
                    res_buy.append(val)

                    if args.eval == 'buy':
                        if val < best:
                            count += 1
                        elif val >= best:
                            count = 0
                            torch.save(model, path_model)
                            best = max(val, best)
                            best_resuls = {'val':result_buy_valid, 'test':result_buy_test}

                    # if args.eval == 'click':
                    #     if val_click < best:
                    #         count += 1
                    #     elif val_click >= best:
                    #         count = 0
                    #         torch.save(model, path_model)
                    #         best = max(val_click, best)
                    #         best_resuls = {'val':result_valid, 'test':result_test}

                    if count >= patience:
                        print('Early stopping!')
                        break
    print(best_resuls)

def test():
    # path_model_test = 'save_tmall/16_Multi_IPS.pt'
    model = torch.load(path_model, map_location='cuda:0').cuda()
    res = TopPerformance('test_buy', model, args.gamma, withratio=False)
    # res = performance_rd('test_buy')
    print(res)


if __name__ == '__main__':
    train()
    # test()




