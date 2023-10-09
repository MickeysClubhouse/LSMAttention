from torch.utils.data import Dataset
import json
import numpy as np
from plan_simple import Plan
from sklearn import preprocessing
import pandas as pd

class MakeTrainingData(Dataset):
    def __init__(self, file_cnt):  # initialize
        self.file_cnt = file_cnt  # number of sst files

    def traversePlan(self, plan):
        nodeType = plan.id.split("_")[0]


def get_operator_enc_dict(ops):
    operators_x = np.array(ops).reshape(len(ops), 1)
    enc = preprocessing.OneHotEncoder()
    op_enc = enc.fit_transform(operators_x).toarray()
    result_dict = {}
    for index, item in enumerate(ops):
        result_dict[item] = list(op_enc[index])
    return result_dict


if __name__ == '__main__':
    operators = ["Selection", "Sort", "HashAgg", "HashJoin", "TableScan", "IndexScan", "TableReader",
                 "IndexReader", "IndexLookUp", "IndexHashJoin"]

    title_regions=[[],[],[],[],[],[]]
    train_json_file = '../data_new/train_plans_scan_pk.json'  # index scan only

    train_plans = []

    # 将文本结果处理成树型结构
    with open(train_json_file, 'r') as f:
        train_cases = json.load(f)
    for case in train_cases:
        train_plans.append(Plan.parse_plan(case['query'], case['plan']))

    operators_enc_dict = get_operator_enc_dict(operators)
    # 对每个scan算子计算模型输入
    for reader_op in train_plans:
        df=pd.read_csv('../data_new/file_info.csv', header=0)
        file_bitmap=reader_op.get_bitmap(df)




