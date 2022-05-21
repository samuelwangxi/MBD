#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/8/10 14:27
# @Author  : name
# @File    : Data.py
import collections

from torch.utils.data import Dataset


class Kwai(Dataset):

    def __init__(self, rt):
        super(Dataset, self).__init__()
        self.uId = list(rt['u_id'])
        self.iId = list(rt['i_id'])
        self.like = list(rt['like'])
        self.click = list(rt['click'])

    def __len__(self):
        return len(self.uId)

    def __getitem__(self, item):
        return self.uId[item], self.iId[item], self.like[item], self.click[item]

class Tmall(Dataset):

    def __init__(self, rt):
        super(Dataset, self).__init__()
        self.uId = list(rt['u_id'])
        self.iId = list(rt['i_id'])
        self.buy = list(rt['buy'])
        self.click = list(rt['click'])

    def __len__(self):
        return len(self.uId)

    def __getitem__(self, item):
        return self.uId[item], self.iId[item], self.buy[item], self.click[item]

class TmallSlot(Dataset):

    def __init__(self, rt):
        super(Dataset, self).__init__()
        self.uId = list(rt['u_id'])
        self.iId = list(rt['i_id'])
        self.buy = list(rt['buy'])
        self.click = list(rt['click'])
        self.pop_tmp = list(rt['pop_tmp'])
        self.pop_pre = list(rt['pop_pre'])

    def __len__(self):
        return len(self.uId)

    def __getitem__(self, item):
        return self.uId[item], self.iId[item], self.buy[item], self.click[item], self.pop_tmp[item],self.pop_pre[item]

def load_user_item_list(train_file):
    train_user_list = collections.defaultdict(list)
    train_item_list = collections.defaultdict(list)
    with open(train_file) as f:
        for line in f.readlines():
            line = line.strip('\n').split(' ')
            if len(line) == 0:
                continue
            line = [int(i) for i in line]
            user = line[0]
            items = line[1:]
            if (len(items) == 0):
                continue
            train_user_list[user] = items
            for item in items:
                train_item_list[item].append(user)
            # n_users = max(n_users, user)
            # n_items = max(n_items, max(items))
    return train_user_list
