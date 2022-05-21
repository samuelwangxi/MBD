import torch
import torch.nn as nn
from torch.nn import Module
from tqdm import trange
from torch.autograd import Variable
import time
import numpy as np

class MACRMF(Module):

    def __init__(self, userNum, itemNum, dim):
        super(MACRMF, self).__init__()
        self.uEmbed = nn.Embedding(userNum, dim)
        self.iEmbed = nn.Embedding(itemNum, dim)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.loss = nn.BCELoss()

        self.Wu = nn.Linear(dim, 1)
        self.Wi = nn.Linear(dim, 1)

        self.cvr = nn.Linear(dim * 2, 64)
        self.cvr1 = nn.Linear(64, 1)

        nn.init.normal_(self.uEmbed.weight, std=0.01)
        nn.init.normal_(self.iEmbed.weight, std=0.01)
    
    def forward(self, userIdx, itemIdx, task):
        uembed = self.uEmbed(userIdx)
        iembed = self.iEmbed(itemIdx)
        
        if task == 'cvr':
            res = self.relu(self.cvr(torch.cat([uembed, iembed], dim=1)))
            res = self.sig(self.cvr1(res))
            
        elif task == 'ctr':
            res = self.sig(torch.sum(torch.mul(uembed, iembed), dim=1))
        
        return res.flatten()

    def do_recommendation(self, users, items, K, c):
        '''
        return topk item index
        '''
        usersEmbed = self.uEmbed(users)
        itemsEmbed = self.iEmbed(items)
        # yk = self.sig(torch.matmul(usersEmbed, itemsEmbed.transpose(1, 0)))
        yu = self.sig(self.Wu(usersEmbed))
        yus = [yu] * len(items)
        yu_cat = torch.cat(yus, dim=1)

        yi = self.sig(self.Wi(itemsEmbed)).transpose(1, 0)
        yis = [yi] * len(users)
        yi_cat = torch.cat(yis, dim=0)
        
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
        yk = torch.mul(ctr, cvr)

        pred = yk * yu_cat * yi_cat - c * yu_cat * yi_cat
        _, topk_index = torch.topk(pred, K)
        return topk_index
    
    def create_bce_loss_like(self, users, pos_items, neg_items, label_like, alpha, beta):
        uembed = self.uEmbed(users)
        iembed = self.iEmbed(pos_items)
        # yk = self.sig(torch.sum(torch.mul(uembed, iembed), dim=1))

        yu = self.sig(self.Wu(uembed)).squeeze()
        yi = self.sig(self.Wi(iembed)).squeeze()
        # pred = yk * yu * yi
        # ctcvr_loss = self.loss(pred, label_like)

        pos_ctr = self.forward(users, pos_items, 'ctr')
        neg_ctr = self.forward(users, neg_items, 'ctr')

        score_cvr = self.forward(users, pos_items, 'cvr')
        score_ctcvr = torch.multiply(pos_ctr, score_cvr)

        target_pos = torch.Tensor([1]*len(users)).cuda()
        target_neg = torch.Tensor([0]*len(users)).cuda()
        
        ctr_loss = self.loss(pos_ctr, target_pos) + self.loss(neg_ctr, target_neg)
        ctcvr_loss = self.loss(score_ctcvr, label_like)
        
        yu_loss = self.loss(yu, label_like)
        yi_loss = self.loss(yi, label_like)

        finalloss = ctcvr_loss + ctr_loss + alpha * yu_loss + beta * yi_loss
        
        # finalloss = ctcvr_loss + alpha * yu_loss + beta * yi_loss
        return finalloss

