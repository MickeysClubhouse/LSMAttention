import json
import os
import sys
from collections import deque
import re
import numpy as np
import pandas as pd
import torch
from plan import Plan, Operator
from torch.utils.data import Dataset
from sklearn import preprocessing
from util import Normalizer

from database_util import *
from database_util import formatFilter, formatJoin, TreeNode, filterDict2Hist


class PlanTreeDataset(Dataset):
    def __init__(self, json_df, train: pd.DataFrame, encoding, hist_file,
                 to_predict, table_sample):

        # final features: embed(operator,join,table,predicate,histogram,sample)
        self.table_sample = table_sample  # sample bitmap
        self.encoding = encoding  # encoding dict
        self.hist_file = hist_file  # histogram

        self.length = len(json_df)
        # train = train.loc[json_df['id']]

        nodes = [plan.root for plan in json_df]  # root nodes
        self.cards = [int(node.act_rows) for node in nodes]  # total rows for plans
        self.costs = [self.extract_exec_time(node.exec_info) for node in nodes]  # total cost for plans

        card_norm = Normalizer(min(self.cards), max(self.cards))
        cost_norm = Normalizer(min(self.costs), max(self.costs))

        # normalize the labels (log of e) with min-max
        self.card_labels = torch.from_numpy(card_norm.normalize_labels(self.cards))
        self.cost_labels = torch.from_numpy(cost_norm.normalize_labels(self.costs))

        self.to_predict = to_predict
        if to_predict == 'cost':
            self.gts = self.costs
            self.labels = self.cost_labels
        elif to_predict == 'card':
            self.gts = self.cards
            self.labels = self.card_labels
        elif to_predict == 'both':  ## try not to use, just in case
            self.gts = self.costs
            self.labels = self.cost_labels
        else:
            raise Exception('Unknown to_predict type')

        idxs = list(range(1, len(nodes) + 1))

        self.treeNodes = []  ## for mem collection

        self.collated_dicts = [self.js_node2dict(i, node) for i, node in
                               zip(idxs, nodes)]  # encode training data one by one

    def extract_exec_time(self, time_str):
        # 定义正则表达式，匹配时间字段和值
        pattern = r'time:\s*(\d+(\.\d+)?)(µs|ms|s)'

        # 在字符串中查找匹配的结果
        match = re.search(pattern, time_str)

        # 如果找到匹配项，则将时间转换为毫秒
        if match:
            time_value = float(match.group(1))
            time_unit = match.group(3)
            if time_unit == "µs":
                time_value /= 1000
            elif time_unit == "s":
                time_value *= 1000
            # time_value = round(time_value, 2)
            return time_value
        else:
            return None

    def js_node2dict(self, idx, node):
        treeNode = self.traversePlan(node, idx, self.encoding)  # 这个node是根节点 这一步还做了encoding的工作
        _dict = self.node2dict(treeNode)  # 每个node的encoding, heights和邻接表
        collated_dict = self.pre_collate(_dict)

        self.treeNodes.clear()
        del self.treeNodes[:]

        return collated_dict

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        return self.collated_dicts[idx], (self.cost_labels[idx], self.card_labels[idx])

    def old_getitem(self, idx):
        return self.dicts[idx], (self.cost_labels[idx], self.card_labels[idx])

    ## pre-process first half of old collator
    def pre_collate(self, the_dict, max_node=30, rel_pos_max=20):

        x = pad_2d_unsqueeze(the_dict['features'], max_node)
        N = len(the_dict['features'])
        attn_bias = torch.zeros([N + 1, N + 1], dtype=torch.float)

        edge_index = the_dict['adjacency_list'].t()
        if len(edge_index) == 0:
            shortest_path_result = np.array([[0]])
            path = np.array([[0]])
            adj = torch.tensor([[0]]).bool()
        else:
            adj = torch.zeros([N, N], dtype=torch.bool)
            adj[edge_index[0, :], edge_index[1, :]] = True

            shortest_path_result = floyd_warshall_rewrite(adj.numpy())

        rel_pos = torch.from_numpy((shortest_path_result)).long()

        attn_bias[1:, 1:][rel_pos >= rel_pos_max] = float('-inf')

        attn_bias = pad_attn_bias_unsqueeze(attn_bias, max_node + 1)
        rel_pos = pad_rel_pos_unsqueeze(rel_pos, max_node)

        heights = pad_1d_unsqueeze(the_dict['heights'], max_node)

        return {
            'x': x,
            'attn_bias': attn_bias,
            'rel_pos': rel_pos,
            'heights': heights
        }

    def node2dict(self, treeNode):

        adj_list, num_child, features = self.topo_sort(treeNode)  # adj_list邻接表,孩子个数,每个node的特征
        heights = self.calculate_height(adj_list, len(features))  # height -> leaf的距离

        return {
            'features': torch.FloatTensor(features),
            'heights': torch.LongTensor(heights),
            'adjacency_list': torch.LongTensor(np.array(adj_list)),

        }

    def topo_sort(self, root_node):
        #        nodes = []
        adj_list = []  # from parent to children
        num_child = []
        features = []

        toVisit = deque()
        toVisit.append((0, root_node))
        next_id = 1
        while toVisit:
            idx, node = toVisit.popleft()
            #            nodes.append(node)
            features.append(node.feature)
            num_child.append(len(node.children))
            for child in node.children:
                toVisit.append((next_id, child))
                adj_list.append((idx, next_id))
                next_id += 1

        return adj_list, num_child, features

    def traversePlan(self, plan, idx, encoding):  # bfs accumulate plan

        nodeType = plan.id.split("_")[0]  # Gather, Seq Scan ...
        typeId = encoding.encode_type(
            nodeType)  # {'Gather': 0, 'Hash Join': 1, 'Seq Scan': 2, 'Hash': 3, 'Bitmap Heap Scan': 4, 'Bitmap Index
        # Scan': 5, 'Nested Loop': 6, 'Index Scan': 7, 'Merge Join': 8, 'Gather Merge': 9, 'Materialize': 10,
        # 'BitmapAnd': 11, 'Sort': 12}
        card = None  # plan['Actual Rows']  may cause bad effect
        filters, alias = formatFilter(plan)
        join = formatJoin(plan)  # join condition
        joinId = encoding.encode_join(join)  # each join condition is in a list
        filters_encoded = encoding.encode_filters(filters, alias)
        #  def __init__(self, nodeType, typeId, filt, card, join, join_str, filterDict):
        root = TreeNode(nodeType, typeId, filters, card, joinId, join, filters_encoded)  # features of root node

        self.treeNodes.append(root)

        if nodeType in ["IndexRangeScan", "TableRowIDScan", "TableFullScan"]:
            root.table = plan.acc_obj
            root.table_id = encoding.encode_table(plan['Relation Name'])
        root.query_id = idx

        root.feature = node2feature(root, encoding, self.hist_file, self.table_sample)  # 'encoding' is the dict
        #    print(root)
        if len(plan.children) > 0:
            for subplan in plan.children:
                subplan.parent = plan
                node = self.traversePlan(subplan, idx, encoding)
                node.parent = root
                root.addChild(node)
        return root

    def calculate_height(self, adj_list, tree_size):
        if tree_size == 1:
            return np.array([0])

        adj_list = np.array(adj_list)
        node_ids = np.arange(tree_size, dtype=int)
        node_order = np.zeros(tree_size, dtype=int)
        uneval_nodes = np.ones(tree_size, dtype=bool)

        parent_nodes = adj_list[:, 0]
        child_nodes = adj_list[:, 1]

        n = 0
        while uneval_nodes.any():
            uneval_mask = uneval_nodes[child_nodes]
            unready_parents = parent_nodes[uneval_mask]

            node2eval = uneval_nodes & ~np.isin(node_ids, unready_parents)
            node_order[node2eval] = n
            uneval_nodes[node2eval] = False
            n += 1
        return node_order


def node2feature(node, encoding, hist_file, table_sample):
    # type, join, filter123, mask123
    # 1, 1, 3x3 (9), 3
    # TODO: add sample (or so-called table)
    num_filter = len(node.filterDict['colId'])
    pad = np.zeros((3, 3 - num_filter))
    filts = np.array(list(node.filterDict.values()))  # cols, ops, vals
    ## 3x3 -> 9, get back with reshape 3,3
    filts = np.concatenate((filts, pad), axis=1).flatten()
    mask = np.zeros(3)
    mask[:num_filter] = 1
    type_join = np.array([node.typeId, node.join])

    hists = filterDict2Hist(hist_file, node.filterDict, encoding)

    # table, bitmap, 1 + 1000 bits
    table = np.array([node.table_id])
    # if node.table_id == 0:
    sample = np.zeros(1000)
    # else:
    #     sample = table_sample[node.query_id][node.table]

    # return np.concatenate((type_join,filts,mask))
    return np.concatenate((type_join, filts, mask, hists, table, sample))


##############  new codespace  ####################
def get_operator_enc_dict(operators):
    operators_x = np.array(operators).reshape(len(operators), 1)
    enc = preprocessing.OneHotEncoder()
    operator_enc = enc.fit_transform(operators_x).toarray()
    result_dict = {}
    for index, item in enumerate(operators):
        result_dict[item] = list(operator_enc[index])
    return result_dict


if __name__ == '__main__':
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
        'mc.id': [1.0, 2609129.0],
        'mc.company_id': [1.0, 234997.0],
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
        'mc.id': 3,
        'mc.company_id': 4,
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
    hist_file = get_hist_file(data_path + 'histogram_string.csv')
    # cost_norm = Normalizer(-3.61192, 12.290855)
    # card_norm = Normalizer(1, 100)
    to_predict = 'cost'

    imdb_path = './imdb/'
    table_sample = get_job_table_sample(imdb_path + 'train')

    train_ds = PlanTreeDataset(train_plans, None, encoding, hist_file, to_predict,
                               table_sample)  # 改了train_cases
