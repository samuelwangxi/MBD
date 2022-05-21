#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/8/10 14:26
# @Author  : name
# @File    : MF.py
import torch
import torch.nn as nn
from torch.nn import Module
from tqdm import trange
from torch.autograd import Variable
import time
import numpy as np


class MF(Module):

    def __init__(self, userNum, itemNum, dim):
        super(MF, self).__init__()
        self.uEmbed = nn.Embedding(userNum, dim)
        self.iEmbed = nn.Embedding(itemNum, dim)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.loss = nn.BCELoss()
        nn.init.normal_(self.uEmbed.weight, std=0.01)
        nn.init.normal_(self.iEmbed.weight, std=0.01)

    def forward(self, userIdx, itemIdx):
        uembed = self.uEmbed(userIdx)
        iembed = self.iEmbed(itemIdx)
        pred = self.sig(torch.sum(torch.mul(uembed, iembed), dim=1))
        return pred.flatten()

    def do_recommendation(self, users, items, K, ratio, gamma, withratio=True):
        '''
        return topk item index
        '''
        usersEmbed = self.uEmbed(users)
        itemsEmbed = self.iEmbed(items)
        predictions = self.sig(torch.matmul(usersEmbed, itemsEmbed.transpose(1, 0)))

        if withratio:
            predictions = torch.mul(predictions, ratio ** gamma)

        _, topk_index = torch.topk(predictions, K)
        return topk_index

    def create_bpr_loss(self, users, pos_items, neg_items, clickgamma, click_neg_gamma, popclick=True):
        users_embed = self.uEmbed(users)
        pos_items_embed = self.iEmbed(pos_items)
        neg_items_embed = self.iEmbed(neg_items)
        pos_scores = torch.sum(torch.multiply(users_embed, pos_items_embed),
                               dim=1)  # users, pos_items, neg_items have the same shape
        neg_scores = torch.sum(torch.multiply(users_embed, neg_items_embed), dim=1)

        if popclick:
            pos_scores = torch.mul(pos_scores, clickgamma)
            neg_scores = torch.mul(neg_scores, click_neg_gamma)

        maxi = torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10)
        mf_loss = torch.negative(torch.mean(maxi))

        return mf_loss

    def create_bce_loss_like(self, users, pos_items, neg_items, label_like, ratiogamma, likegammal, withratio=True,
                             poplike=True):
        score_cvr = self.forward(users, pos_items)
        if withratio:
            score_cvr = torch.mul(score_cvr, ratiogamma)
        if poplike:
            score_cvr = torch.mul(score_cvr, likegammal)
        cvr_loss = self.loss(score_cvr, label_like)

        return cvr_loss


class MF_multi(Module):

    def __init__(self, userNum, itemNum, dim, loss):
        super(MF_multi, self).__init__()
        self.uEmbed = nn.Embedding(userNum, dim)
        self.iEmbed = nn.Embedding(itemNum, dim)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()

        self.cvr = nn.Linear(dim * 2, 64)
        self.cvr1 = nn.Linear(64, 1)

        self.imp = nn.Linear(dim * 2, 64)
        self.imp1 = nn.Linear(64, 1)

        self.loss = nn.BCELoss()
        self.L = loss

        nn.init.normal_(self.uEmbed.weight, std=0.01)
        nn.init.normal_(self.iEmbed.weight, std=0.01)

    def forward(self, userIdx, itemIdx, task):
        uembed = self.uEmbed(userIdx)
        iembed = self.iEmbed(itemIdx)

        if self.L == 'multi':
            if task == 'cvr':
                res = self.relu(self.cvr(torch.cat([uembed, iembed], dim=1)))
                res = self.sig(self.cvr1(res))

            elif task == 'ctr':
                res = self.sig(torch.sum(torch.mul(uembed, iembed), dim=1))

        if self.L == 'IPS+DR':
            if task == 'cvr':
                res = self.relu(self.cvr(torch.cat([uembed, iembed], dim=1)))
                res = self.sig(self.cvr1(res))

            elif task == 'ctr':
                res = self.sig(torch.sum(torch.mul(uembed, iembed), dim=1))


            elif task == 'imp':
                res = self.relu(self.imp(torch.cat([uembed, iembed], dim=1)))
                res = abs(self.sig(self.imp1(res)))

        return res.flatten()

    def do_recommendation(self, users, items, K, loss, ratio, withratio=True):
        '''
        return topk item index
        # '''
        if loss == 'IPS+DR':
            ctr, cvr = [], []
            users = users.cpu().numpy()
            for u in range(len(users)):
                user = [users[u]] * len(items)
                ct = self.forward(torch.LongTensor(user).cuda(), items, 'ctr')
                cv = self.forward(torch.LongTensor(user).cuda(), items, 'cvr')
                ctr.append(ct.unsqueeze(0))
                cvr.append(cv.unsqueeze(0))
            ctr = torch.cat(ctr, dim=0)
            cvr = torch.cat(cvr, dim=0)
            # predictions = cvr
            predictions = torch.mul(ctr, cvr)

        if loss == 'multi':
            ctr, cvr = [], []
            users = users.cpu().numpy()
            for u in range(len(users)):
                user = [users[u]] * len(items)
                ct = self.forward(torch.LongTensor(user).cuda(), items, 'ctr')
                cv = self.forward(torch.LongTensor(user).cuda(), items, 'cvr')
                ctr.append(ct.unsqueeze(0))
                cvr.append(cv.unsqueeze(0))
            ctr = torch.cat(ctr, dim=0)
            cvr = torch.cat(cvr, dim=0)
            predictions = torch.mul(ctr, cvr)

        if withratio:
            predictions = torch.mul(predictions, ratio)

        _, topk_index = torch.topk(predictions, K)
        return topk_index

    def do_recommendation_abl(self, users, items, K, loss, ratio, likegamma, withratio=True, poplike=True):
        '''
        return topk item index
        # '''
        if loss == 'IPS+DR':
            ctr, cvr = [], []
            users = users.cpu().numpy()
            for u in range(len(users)):
                user = [users[u]] * len(items)
                ct = self.forward(torch.LongTensor(user).cuda(), items, 'ctr')
                cv = self.forward(torch.LongTensor(user).cuda(), items, 'cvr')
                ctr.append(ct.unsqueeze(0))
                cvr.append(cv.unsqueeze(0))
            ctr = torch.cat(ctr, dim=0)
            cvr = torch.cat(cvr, dim=0)
            # predictions = cvr
            predictions = torch.mul(ctr, cvr)

        if loss == 'multi':
            ctr, cvr = [], []
            users = users.cpu().numpy()
            for u in range(len(users)):
                user = [users[u]] * len(items)
                ct = self.forward(torch.LongTensor(user).cuda(), items, 'ctr')
                cv = self.forward(torch.LongTensor(user).cuda(), items, 'cvr')
                ctr.append(ct.unsqueeze(0))
                cvr.append(cv.unsqueeze(0))
            ctr = torch.cat(ctr, dim=0)
            cvr = torch.cat(cvr, dim=0)
            predictions = torch.mul(ctr, cvr)

        if withratio:
            predictions = torch.mul(predictions, ratio)

        if poplike:
            predictions = torch.mul(predictions, likegamma)

        _, topk_index = torch.topk(predictions, K)
        return topk_index

    def create_esmm_loss(self, users, pos_items, neg_items, label_like, ratiogamma, likegammal, clickgamma,
                         click_neg_gamma,
                         withratio=False, poplike=False, popclick=False):

        pos_ctr = self.forward(users, pos_items, 'ctr')
        neg_ctr = self.forward(users, neg_items, 'ctr')

        if popclick:
            pos_ctr = torch.mul(pos_ctr, clickgamma)
            neg_ctr = torch.mul(neg_ctr, click_neg_gamma)

        score_cvr = self.forward(users, pos_items, 'cvr')
        score_ctcvr = torch.multiply(pos_ctr, score_cvr)

        if poplike:
            score_ctcvr = torch.mul(score_ctcvr, likegammal)

        if withratio:
            score_ctcvr = torch.mul(score_ctcvr, ratiogamma)

        target_pos = torch.Tensor([1] * len(users)).cuda()
        target_neg = torch.Tensor([0] * len(users)).cuda()

        ctr_loss = self.loss(pos_ctr, target_pos) + self.loss(neg_ctr, target_neg)
        ctcvr_loss = self.loss(score_ctcvr, label_like)
        finalloss = ctcvr_loss + ctr_loss
        return finalloss

    def create_imp_loss(self, users, pos_items, neg_items, label_like):
        pos_ctr = self.forward(users, pos_items, 'ctr')
        neg_ctr = self.forward(users, neg_items, 'ctr')

        target_pos = torch.Tensor([1] * len(users)).cuda()
        target_neg = torch.Tensor([0] * len(users)).cuda()

        ctr_loss = self.loss(pos_ctr, target_pos) + self.loss(neg_ctr, target_neg)

        score_cvr = self.forward(users, pos_items, 'cvr')
        loss_imp_pos = self.forward(users, pos_items, 'imp')
        loss_imp_neg = self.forward(users, neg_items, 'imp')

        # multi_IPS 
        # v = 1e-3
        # M = torch.Tensor([[v]] * len(users)).cuda()
        # pos_scores, index = torch.max(torch.cat([pos_scores.unsqueeze(1), M], dim=1), 1)

        r = 0.90
        P = 1 / pos_ctr
        t = (torch.mean(P) - torch.min(P)) * r + torch.min(P)
        M = torch.Tensor([[t]] * len(users)).cuda()
        P, index = torch.min(torch.cat([P.unsqueeze(1), M], dim=1), 1)

        cvr_loss = - label_like * torch.log(score_cvr + 1e-5) - (1 - label_like) * torch.log((1 - score_cvr) + 1e-5)
        IPS_loss = torch.mean(cvr_loss * P)

        DR_loss = torch.mean(abs(cvr_loss - loss_imp_pos) * P + loss_imp_neg)

        final_loss = ctr_loss + IPS_loss

        return final_loss


class PDA(Module):
    def __init__(self, userNum, itemNum, dim):
        super(PDA, self).__init__()
        self.uEmbed = nn.Embedding(userNum, dim)
        self.iEmbed = nn.Embedding(itemNum, dim)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()

        self.cvr = nn.Linear(dim * 2, 64)
        self.cvr1 = nn.Linear(64, 1)

        # self.elu = torch.nn.functional.elu()
        self.loss = nn.BCELoss()

        nn.init.normal_(self.uEmbed.weight, std=0.01)
        nn.init.normal_(self.iEmbed.weight, std=0.01)

    def forward(self, userIdx, itemIdx, task):
        uembed = self.uEmbed(userIdx)
        iembed = self.iEmbed(itemIdx)

        if task == 'cvr':
            res = self.relu(self.cvr(torch.cat([uembed, iembed], dim=1)))
            res = self.sig(self.cvr1(res))

        elif task == 'ctr':
            # res = torch.nn.functional.elu((torch.sum(torch.mul(uembed, iembed), dim=1)))
            res = self.sig(torch.sum(torch.mul(uembed, iembed), dim=1))

        return res.flatten()

    def do_recommendation(self, users, items, K, pop, likegammal):
        '''
        return topk item index
        # '''
        ctr, cvr = [], []
        users = users.cpu().numpy()
        for u in range(len(users)):
            user = [users[u]] * len(items)
            ct = self.forward(torch.LongTensor(user).cuda(), items, 'ctr')
            cv = self.forward(torch.LongTensor(user).cuda(), items, 'cvr')
            ctr.append(ct.unsqueeze(0))
            cvr.append(cv.unsqueeze(0))
        ctr = torch.cat(ctr, dim=0)
        cvr = torch.cat(cvr, dim=0)
        predictions = torch.mul(ctr, cvr)

        predictions = torch.mul(predictions, pop ** likegammal)
        _, topk_index = torch.topk(predictions, K)
        return topk_index

    def pda_loss(self, users, pos_items, neg_items, likegammal, label_like, poplike=False):

        # pos_ctr = self.forward(users, pos_items, 'ctr')
        # neg_ctr = self.forward(users, neg_items, 'ctr')
        #
        # score_cvr = self.forward(users, pos_items, 'cvr')
        # score_ctcvr = torch.multiply(pos_ctr, score_cvr)
        #
        # score_cvr_neg = self.forward(users, neg_items, 'cvr')
        # score_ctcvr_neg = torch.multiply(neg_ctr, score_cvr)
        #
        # if poplike:
        #     score_ctcvr = torch.mul(score_ctcvr, likegammal)
        #
        # # target_pos = torch.Tensor([1]*len(users)).cuda()
        # # target_neg = torch.Tensor([0]*len(users)).cuda()
        #
        # # ctr_loss = self.loss(pos_ctr, target_pos) + self.loss(neg_ctr, target_neg)
        # # ctcvr_loss = self.loss(score_ctcvr, label_like)
        #
        # ctr_loss = torch.log(self.sig(pos_ctr - neg_ctr)+1e-10)
        # ctcvr_loss = torch.log(self.sig(score_ctcvr - score_ctcvr_neg)+1e-10)
        # finalloss = ctcvr_loss + ctr_loss
        # # finalloss = - torch.mean(finalloss)
        # return finalloss

        pos_ctr = self.forward(users, pos_items, 'ctr')
        neg_ctr = self.forward(users, neg_items, 'ctr')

        score_cvr = self.forward(users, pos_items, 'cvr')
        score_ctcvr = torch.multiply(pos_ctr, score_cvr)

        if poplike:
            score_ctcvr = torch.mul(score_ctcvr, likegammal)

        target_pos = torch.Tensor([1] * len(users)).cuda()
        target_neg = torch.Tensor([0] * len(users)).cuda()

        ctr_loss = self.loss(pos_ctr, target_pos) + self.loss(neg_ctr, target_neg)
        ctcvr_loss = self.loss(score_ctcvr, label_like)
        finalloss = ctcvr_loss + ctr_loss
        return finalloss
