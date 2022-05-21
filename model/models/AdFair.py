import torch
import torch.nn as nn
import logging
import torch.nn.functional as F
import os
from torch.nn import Module
from tqdm import trange
from torch.autograd import Variable
import time
import numpy as np


class AdFair(Module):

    def __init__(self, userNum, itemNum, dim, feature_num, lamb=1, filter_mode='combine'):
        super(AdFair, self).__init__()
        self.uEmbed = nn.Embedding(userNum, dim)
        self.iEmbed = nn.Embedding(itemNum, dim)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.loss = nn.BCELoss()
        self.dim = dim
        self.feature_num = feature_num
        self.filter_mode = filter_mode
        self.optimizer = None

        nn.init.normal_(self.uEmbed.weight, std=0.01)
        nn.init.normal_(self.iEmbed.weight, std=0.01)

        self.cvr = nn.Linear(dim * 2, 64)
        self.cvr1 = nn.Linear(64, 1)

        self.isequential = nn.Sequential(
            # nn.Linear(dim, int(dim/2)),
            # nn.LeakyReLU(),
            # nn.Dropout(p=0.1),
            # nn.Linear(int(dim/2), dim),
            # nn.LeakyReLU(),
            # nn.Dropout(p=0.1),
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.BatchNorm1d(dim),
        )

        self.filter_num = self.feature_num if filter_mode == 'combine' else 2 ** feature_num
        # self.filters = nn.ModuleList(
        #     [self.sequential for i in range(self.filter_num)]
        #     )

    def apply_filter(self, vectors, filter_mask):
        if self.filter_mode == 'separate':
            filter_mask = np.asarray(filter_mask)
            idx = filter_mask.dot(2 ** np.arange(filter_mask.size))
            sens_filter = self.filter_dict[str(idx)]
            result = sens_filter(vectors)
        elif self.filter_mode == 'combine':
            # results = [self.filters[i](vectors) * filter_mask[i] for i in range(self.feature_num)]
            # results = np.sum(results) / self.feature_num
            results = self.isequential(vectors)

        return results

    def forward(self, userIdx, itemIdx, task):
        uembed = self.uEmbed(userIdx)
        iembed = self.iEmbed(itemIdx)
        # iembed = self.isequential(iembed)

        if task == 'cvr':
            res = self.relu(self.cvr(torch.cat([uembed, iembed], dim=1)))
            res = self.sig(self.cvr1(res))

        elif task == 'ctr':
            res = self.sig(torch.sum(torch.mul(uembed, iembed), dim=1))

        return res.flatten()

    def do_recommendation(self, users, items, K):
        '''
        return topk item index
        '''
        # uembed = self.uEmbed(users)
        # iembed = self.iEmbed(items)
        # mask = torch.FloatTensor([[1]] * len(items)).cuda()
        # # ifilterEmbed = self.apply_filter(iembed, mask)
        # iembed = self.isequential(iembed)
        # pred = self.sig(torch.matmul(uembed, iembed.transpose(1, 0)))

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
        pred = torch.mul(ctr, cvr)

        _, topk_index = torch.topk(pred, K)
        return topk_index

    def create_loss(self, users, pos_items, neg_items, label_like):

        pos_ctr = self.forward(users, pos_items, 'ctr')
        neg_ctr = self.forward(users, neg_items, 'ctr')

        score_cvr = self.forward(users, pos_items, 'cvr')
        score_ctcvr = torch.multiply(pos_ctr, score_cvr)

        target_pos = torch.Tensor([1] * len(users)).cuda()
        target_neg = torch.Tensor([0] * len(users)).cuda()

        ctr_loss = self.loss(pos_ctr, target_pos) + self.loss(neg_ctr, target_neg)
        ctcvr_loss = self.loss(score_ctcvr, label_like)
        finalloss = ctcvr_loss + ctr_loss
        return finalloss


class Discriminator(Module):

    def __init__(self, embed_dim, dropout=0.3):
        super(Discriminator, self).__init__()
        self.embed_dim = int(embed_dim)
        self.dropout = dropout
        self.loss = nn.MSELoss()
        self.neg_slope = 0.2
        self.out_dim = 1
        self.optimizer = None

        self.network = nn.Sequential(
            nn.Linear(int(self.embed_dim), int(self.embed_dim / 2), bias=True),
            nn.LeakyReLU(self.neg_slope),
            nn.Dropout(p=self.dropout),
            nn.Linear(int(self.embed_dim / 2), self.out_dim, bias=True),
            nn.Sigmoid()
        )

    @staticmethod
    def init_weights(m):
        """
        initialize nn weights，called in main.py
        :param m: parameter or the nn
        :return:
        """
        if type(m) == torch.nn.Linear:
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)
            if m.bias is not None:
                torch.nn.init.normal_(m.bias, mean=0.0, std=0.01)
        elif type(m) == torch.nn.Embedding:
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)

    def forward(self, embeddings, labels):
        scores = self.network(embeddings).squeeze()
        loss = self.loss(scores, labels)
        return loss
