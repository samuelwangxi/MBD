#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/8/10 16:13
# @Author  : name
# @File    : data_process_like.py

# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/8/10 10:03
# @Author  : name
# @File    : data_process.py
import pandas as pd
import numpy as np
import time
import pickle
import os
from tqdm import trange

path = 'data_time'
data = pd.read_csv("train_interaction_kwai.txt", header=None, sep='\t')
data.columns = ['u_id', 'photo_id', 'click', 'like', 'follow', 'time', 'play_time', 'duration_time']
# print(data.head())
data = data[data['click'] != 0]
# print(data.head())
data = data[['u_id', 'photo_id', 'click', 'like', 'follow', 'time']]


def filter_g_k_one(data, k=10, u_name='user_id', i_name='business_id', y_name='stars'):
    item_group = data.groupby(i_name).agg({y_name: 'count'})
    print(item_group.head())
    item_g10 = item_group[item_group[y_name] >= k].index
    print(item_g10)
    data_new = data[data[i_name].isin(item_g10)]
    user_group = data_new.groupby(u_name).agg({y_name: 'count'})
    user_g10 = user_group[user_group[y_name] >= k].index
    data_new = data_new[data_new[u_name].isin(user_g10)]
    return data_new


def filter_tot(data, k=10, u_name='user_id', i_name='business_id', y_name='stars'):
    data_new = data
    while True:
        data_new = filter_g_k_one(data_new, k=k, u_name=u_name, i_name=i_name, y_name=y_name)
        m1 = data_new.groupby(i_name).agg({y_name: 'count'})
        m2 = data_new.groupby(u_name).agg({y_name: 'count'})
        num1 = m1[y_name].min()
        num2 = m2[y_name].min()
        print('item min:', num1, 'user min:', num2)
        if num1 >= k and num2 >= k:
            break
    return data_new

def like_filter(data, k = 1):
    data_like = data.groupby(['photo_id'])['like'].sum().reset_index(name="like")
    data_like_item = data_like[data_like['like'] >= k]

    data_like = data.groupby(['u_id'])['like'].sum().reset_index(name="like")
    data_like_user = data_like[data_like['like'] >= k]

    photo_ids = list(data_like_item['photo_id'].values)
    user_ids = list(data_like_user['u_id'].values)

    data_new = data[data['photo_id'].isin(photo_ids) & data['u_id'].isin(user_ids)]
    return data_new

data_10core = like_filter(data, 5)

data_10core = filter_tot(data_10core, k=10, u_name='u_id', i_name='photo_id', y_name='like')
data_like = data_10core[data_10core['like'] != 0]

print('data_like num:', len(data_like), 'data_click:', len(data_10core), 'percent:', len(data_like)/len(data_10core))
# data_10core = pd.read_csv('kwai_data_10_cores_.csv', sep='\t')
# data_10core = data
# data_10core.columns = ['u_id', 'photo_id', 'click', 'like', 'follow', 'time']
user_num = data_10core['u_id'].unique().shape[0]
item_num = data_10core['photo_id'].unique().shape[0]
itr_num = data_10core.shape[0] * 1.
sparse = itr_num / (user_num * item_num)
print("user:", user_num, 'item_num:', item_num, 'sparse:', sparse, 'sparse_like', sparse * len(data_like)/len(data_10core))
print("avg itr of user:", itr_num / user_num, 'avg itr of item:', itr_num / item_num)

data = data_10core
data = data.sort_values(by=['time'])
data.to_csv(os.path.join(path, 'kwai_data_10_cores.csv'), index=False, sep='\t')

slot_num = 10
time_span = (data['time'].max() - data['time'].min()) // slot_num
time_min = data['time'].min()
data['time_slot'] = data['time'].apply(lambda x: min((x - time_min) // time_span, slot_num - 1))
data.tail(5)

# data.to_csv('data_timeslot.csv', sep='\t', index=False)
# print('data with slot saved!')
alpha = 0.5
# def time_pop(data):
#     iid = list(data['photo_id'].unique())
#     dict_pop = {}
#
#     iids = list(data['photo_id'].values)
#     likes = list(data['like'].values)
#     click = list(data['click'].values)
#     slots = list(data['time_slot'].values)
#
#     for i in trange(len(iids)):
#         if iids[i] in dict_pop.keys():
#             dict_pop[iids[i]][slots[i]] += 1
#         else:
#             dict_pop[iids[i]] = [0 for _ in range(10)]
#     pop_valid = []
#     pop_test = []
#     ids = []
#     all_pop = []
#     for k,v in dict_pop.items():
#         pop_valid.append(max(v[8] + alpha * (v[8] - v[7]), 0))
#         pop_test.append(max(v[9] + alpha * (v[9] - v[8]), 0))
#         ids.append(k)
#         all_pop.append(v)
#     pop_valid = 1 + np.around((pop_valid - np.min(pop_valid)) / (np.max(pop_valid) - np.min(pop_valid)), decimals=4)
#     pop_test = 1 + np.around((pop_test - np.min(pop_test)) / (np.max(pop_test) - np.min(pop_test)), decimals=4)
#     df = pd.DataFrame({'photo_id':ids, 'pop_valid':pop_valid, 'pop_test':pop_test, 'all_pop':all_pop})
#
#     return df

# item_pop = time_pop(data)
# print(item_pop.head())

train_slot = round(slot_num * 0.8)
data_train = data[data['time_slot'].isin(list(range(0, train_slot)))]
data_valid = data[data['time_slot'].isin(list(range(8, 9)))]
data_test = data[data['time_slot'].isin(list(range(9, 10)))]
print("train number:", data_train.shape, "data_test:", data_test.shape, 'data_valid:', data_valid.shape)

user_in_train = data_train['u_id'].unique()
item_in_train = data_train['photo_id'].unique()

data_test = data_test[data_test['u_id'].isin(user_in_train)]
print("user not include in user_items_test:", data_test.shape)
data_test = data_test[data_test['photo_id'].isin(item_in_train)]
print("train:", data_train.shape[0], 'not-new test:', data_test.shape[0])

data_valid = data_valid[data_valid['u_id'].isin(user_in_train)]
print("user not include in user_items_valid:", data_valid.shape)
data_valid = data_valid[data_valid['photo_id'].isin(item_in_train)]
print("train:", data_train.shape[0], 'not-new valid:', data_valid.shape[0])

data_train = data_train.drop_duplicates(subset=['u_id', 'photo_id'], keep='first')
data_test = data_test.drop_duplicates(subset=['u_id', 'photo_id'], keep='first')
data_valid = data_valid.drop_duplicates(subset=['u_id', 'photo_id'], keep='first')
print("not repeat train:", data_train.shape[0], 'not-repeat test:', data_test.shape[0], 'not-repaet valid:', data_valid.shape[0])
print("user in train:", user_in_train.shape, 'item in train:', item_in_train.shape)

# for inter
user = data_train['u_id'].unique()
item = data_train['photo_id'].unique()
user_to_id = dict(zip(list(user), list(np.arange(user.shape[0]))))
with open(os.path.join(path, 'user2id_dict.pkl'), 'wb') as file0:
    pickle.dump(user_to_id, file0)

item_to_id = dict(zip(list(item), list(range(item.shape[0]))))
id2item = dict(zip(list(range(item.shape[0])), list(item)))
with open(os.path.join(path, 'item2id_dict.pkl'), 'wb') as file1:
    pickle.dump(item_to_id, file1)
with open(os.path.join(path, 'id2item_dict.pkl'), 'wb') as file2:
    pickle.dump(id2item, file2)

print("user num:", user.shape)
print("item num:", item.shape)

data['uid'] = data['u_id'].map(user_to_id)
data['iid'] = data['photo_id'].map(item_to_id)
data = data[['uid', 'iid', 'time_slot', 'like', 'click', 'pop_valid', 'pop_test']]
data = data[['iid', 'pop_valid', 'pop_test']]
data = data.drop_duplicates(['iid', 'pop_valid', 'pop_test'])
# item_pop['iid'] = item_pop['photo_id'].map(item_to_id)
# item_pop = item_pop[['iid', 'pop_valid', 'pop_test', 'all_pop']]
# item_pop.to_csv('kwai_item_id_pop2.csv', sep='\t', index=False)

data_train['uid'] = data_train['u_id'].map(user_to_id)
data_train['iid'] = data_train['photo_id'].map(item_to_id)

data_test['uid'] = data_test['u_id'].map(user_to_id)
data_test['iid'] = data_test['photo_id'].map(item_to_id)

data_valid['uid'] = data_valid['u_id'].map(user_to_id)
data_valid['iid'] = data_valid['photo_id'].map(item_to_id)
data_test.head(2)

# # # tot stage data
# data_train = data_train[['uid', 'iid', 'time_slot', 'like', 'click', 'pop_tmp', 'pop_pre']]
# data_test = data_test[['uid', 'iid', 'time_slot', 'like', 'click', 'pop_tmp', 'pop_pre']]
# data_valid = data_valid[['uid', 'iid', 'time_slot', 'like', 'click', 'pop_tmp', 'pop_pre']]

data_train = data_train[['uid', 'iid', 'time_slot', 'like', 'click']]
data_test = data_test[['uid', 'iid', 'time_slot', 'like', 'click']]
data_valid = data_valid[['uid', 'iid', 'time_slot', 'like', 'click']]

# mm = pd.concat([data_train, data_test], axis=0)
# mm.head(4)
# mm.tail(5)
# mm = mm[['uid', 'iid', 'like', 'time_slot']]
# mm.columns = ['user_id:token', 'item_id:token', 'rating:float', 'timestamp:float']
# mm.head(5)
# mm.to_csv(os.path.join(path,"kwai.inter"), index=False, sep=' ')
# m = data_test.groupby('uid').agg({'iid': 'count'})
# m.head(2)
# m.describe()

# for click
# data_valid = data_train.sample(frac=0.1, replace=False)
# data_train = data_train.drop(data_valid.index)

data_train_like = data_train[data_train['like']==1]
data_test_like = data_test[data_test['like']==1]
data_valid_like = data_valid[data_valid['like']==1]

# # saving ...
data_train.to_csv(os.path.join(path, 'train_interaction_filter.txt'), header=False, index=False, sep=' ')
data_valid.to_csv(os.path.join(path, 'valid_interaction_filter.txt'), header=None, index=False, sep=' ')
data_test.to_csv(os.path.join(path, 'test_interaction_filter.txt'), header=None, index=False, sep=' ')

data_train_like.to_csv(os.path.join(path, 'train_like_interaction_filter.txt'), header=False, index=False, sep=' ')
data_valid_like.to_csv(os.path.join(path, 'valid_like_interaction_filter.txt'), header=None, index=False, sep=' ')
data_test_like.to_csv(os.path.join(path, 'test_like_interaction_filter.txt'), header=None, index=False, sep=' ')


user_items_train = data_train.sort_values(by='uid')
train_itr = user_items_train.values[:, 0:2]

with open(os.path.join(path, 'train.txt'), 'w') as f:
    u_pre = train_itr[0, 0]
    k = 0
    for x in train_itr:
        u = int(x[0])
        i = int(x[1])
        if u != u_pre or k == 0:
            u_pre = u
            if k > 0:
                f.write('\n')
            f.write(str(u))
            k = 1
        f.write(' ' + str(i))

user_items_test = data_test.sort_values(by='uid')
test_itr = user_items_test.values[:, 0:2]
#
with open(os.path.join(path, 'test.txt'), 'w') as f:
    u_pre = test_itr[0, 0]
    k = 0
    for x in test_itr:
        u = int(x[0])
        i = int(x[1])
        if u != u_pre or k == 0:
            u_pre = u
            if k > 0:
                f.write('\n')
            f.write(str(u))
            k = 1
        f.write(' ' + str(i))

user_items_valid = data_valid.sort_values(by='uid')
valid_itr = user_items_valid.values[:, 0:2]
#
with open(os.path.join(path, 'valid.txt'), 'w') as f:
    u_pre = valid_itr[0, 0]
    k = 0
    for x in valid_itr:
        u = int(x[0])
        i = int(x[1])
        if u != u_pre or k == 0:
            u_pre = u
            if k > 0:
                f.write('\n')
            f.write(str(u))
            k = 1
        f.write(' ' + str(i))


user_like_train = data_train_like.sort_values(by='uid')
train_like_itr = user_like_train.values[:, 0:2]

with open(os.path.join(path, 'train_like.txt'), 'w') as f:
    u_pre = train_like_itr[0, 0]
    k = 0
    for x in train_like_itr:
        u = int(x[0])
        i = int(x[1])
        if u != u_pre or k == 0:
            u_pre = u
            if k > 0:
                f.write('\n')
            f.write(str(u))
            k = 1
        f.write(' ' + str(i))

user_like_test = data_test_like.sort_values(by='uid')
test_like_itr = user_like_test.values[:, 0:2]
#
with open(os.path.join(path, 'test_like.txt'), 'w') as f:
    u_pre = test_like_itr[0, 0]
    k = 0
    for x in test_like_itr:
        u = int(x[0])
        i = int(x[1])
        if u != u_pre or k == 0:
            u_pre = u
            if k > 0:
                f.write('\n')
            f.write(str(u))
            k = 1
        f.write(' ' + str(i))

user_like_valid = data_valid_like.sort_values(by='uid')
valid_like_itr = user_like_valid.values[:, 0:2]
#
with open(os.path.join(path, 'valid_like.txt'), 'w') as f:
    u_pre = valid_like_itr[0, 0]
    k = 0
    for x in valid_like_itr:
        u = int(x[0])
        i = int(x[1])
        if u != u_pre or k == 0:
            u_pre = u
            if k > 0:
                f.write('\n')
            f.write(str(u))
            k = 1
        f.write(' ' + str(i))

print(len(data_train), len(data_test), len(data_valid))
print(len(data_train_like), len(data_test_like), len(data_valid_like))
