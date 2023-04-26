import numpy as np
import os
import torch
import torch.nn as nn
import time
import pandas as pd
from scipy.stats import pearsonr

from model.util import Normalizer
from model.database_util import get_hist_file, get_job_table_sample, collator
from model.model import QueryFormer
from model.database_util import Encoding
from feature_extraction import PlanTreeDataset
from model.trainer import eval_workload, train
import json
import os
import sys
from collections import deque
import re

from plan import Plan, Operator
from torch.utils.data import Dataset
from sklearn import preprocessing
from util import Normalizer

from database_util import *
from database_util import formatFilter, formatJoin, TreeNode, filterDict2Hist

from model.util import seed_everything

from feature_extraction import get_operator_enc_dict

seed_everything()


class Args:
    bs = 1024
    lr = 0.001
    epochs = 200
    clip_size = 50
    embed_size = 64
    pred_hid = 128
    ffn_dim = 128
    head_size = 12
    n_layers = 8
    dropout = 0.1
    sch_decay = 0.6
    device = 'cuda:0'
    newpath = './results/full/cost/'
    to_predict = 'cost'


args = Args()

if not os.path.exists(args.newpath):
    os.makedirs(args.newpath)

model = QueryFormer(emb_size=args.embed_size, ffn_dim=args.ffn_dim, head_size=args.head_size, dropout=args.dropout,
                    n_layers=args.n_layers, use_sample=True, use_hist=True, pred_hid=args.pred_hid)

_ = model.to(args.device)
to_predict = 'cost'

operators = ["Projection", "Selection", "Sort", "HashAgg", "HashJoin", "TableScan", "IndexScan", "TableReader",
             "IndexReader", "IndexLookUp", "IndexHashJoin"]

train_json_file = 'data/train_plans_serial.json'  # serial
test_json_file = 'data/train_plans_serial.json'
train_plans, test_plans = [], []

with open(train_json_file, 'r') as f:
    train_cases = json.load(f)
for case in train_cases:
    train_plans.append(Plan.parse_plan(case['query'], case['plan']))

with open(test_json_file, 'r') as f:
    test_cases = json.load(f)
for case in test_cases:
    test_plans.append(Plan.parse_plan(case['query'], case['plan']))

operators_enc_dict = get_operator_enc_dict(operators)

min_max = {
    't.id': [1.0, 2528312.0],
    't.kind_id': [1.0, 7.0],
    't.production_year': [1880.0, 2019.0],
    'title.episode_of_id': [0.0, 2528186.0],
    'title.season_nr': [0.0, 2013.0],
    'mc.movie_id': [2.0, 2525745.0],
    'mc.company_type_id': [1.0, 2.0],
    'ci.id': [1.0, 36244344.0],
    'ci.movie_id': [1.0, 2525975.0],
    'ci.person_id': [1.0, 4061926.0],
    'ci.role_id': [1.0, 11.0],
    'mi.id': [1.0, 14835720.0],
    'mi.movie_id': [1.0, 2526430.0],
    'mi.info_type_id': [1.0, 110.0],
    'mi_idx.id': [1.0, 1380035.0],
    'mi_idx.movie_id': [2.0, 2525793.0],
    'mi_idx.info_type_id': [99.0, 113.0],
    'mk.id': [1.0, 4523930.0],
    'mk.movie_id': [2.0, 2525971.0],
    'mk.keyword_id': [1.0, 134170.0]
}

col2idx = {
    't.id': 0,
    't.kind_id': 1,
    't.production_year': 2,
    'title.episode_of_id': 3,
    'title.season_nr': 4,
    'mc.movie_id': 5,
    'mc.company_type_id': 6,
    'ci.id': 7,
    'ci.movie_id': 8,
    'ci.person_id': 9,
    'ci.role_id': 10,
    'mi.id': 11,
    'mi.movie_id': 12,
    'mi.info_type_id': 13,
    'mi_idx.id': 14,
    'mi_idx.movie_id': 15,
    'mi_idx.info_type_id': 16,
    'mk.id': 17,
    'mk.movie_id': 18,
    'mk.keyword_id': 19,
    'NA': 20
}

encoding = Encoding(min_max, col2idx)

# old
data_path = "data/"

hist_file = get_histograms()
# cost_norm = Normalizer(-3.61192, 12.290855)
# card_norm = Normalizer(1, 100)
to_predict = 'cost'

imdb_path = './imdb/'
table_sample = get_job_table_sample(imdb_path + 'train')

train_ds = PlanTreeDataset(train_plans, None, encoding, hist_file, to_predict, table_sample)  # 改了train_cases
val_ds = PlanTreeDataset(train_plans, None, encoding, hist_file, to_predict, table_sample)

cost_norm = Normalizer(0.1157, 3990.0)


crit = nn.MSELoss()
model, best_path = train(model, train_ds, val_ds, crit, cost_norm, args)

print("done")