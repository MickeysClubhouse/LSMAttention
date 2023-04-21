import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from plan import Operator, Plan
from plan import Operator, Plan
from sklearn import preprocessing
import numpy as np
import json

operators = ["Projection", "Selection", "Sort", "HashAgg", "HashJoin", "TableScan", "IndexScan", "TableReader",
             "IndexReader", "IndexLookUp", "IndexHashJoin"]


def get_operator_enc_dict(operators):
    operators_x = np.array(operators).reshape(len(operators), 1)
    enc = preprocessing.OneHotEncoder()
    operator_enc = enc.fit_transform(operators_x).toarray()
    result_dict = {}
    for index, item in enumerate(operators):
        result_dict[item] = list(operator_enc[index])
    return result_dict


operators_enc_dict = get_operator_enc_dict(operators)


# There are many ways to extract features from plan:
# 1. The simplest way is to extract features from each node and sum them up. For example, we can get
#      a  the number of nodes;
#      a. the number of occurrences of each operator;
#      b. the sum of estRows for each operator.
#    However we lose the tree structure after extracting features.
# 2. The second way is to extract features from each node and concatenate them in the DFS traversal order.
#                  HashJoin_1
#                  /          \
#              IndexJoin_2   TableScan_6
#              /         \
#          IndexScan_3   IndexScan_4
#    For example, we can concatenate the node features of the above plan as follows:
#    [Feat(HashJoin_1)], [Feat(IndexJoin_2)], [Feat(IndexScan_3)], [Padding], [Feat(IndexScan_4)], [Padding], [Padding], [Feat(TableScan_6)], [Padding], [Padding]
#    Notice1: When we traverse all the children in DFS, we insert [Padding] as the end of the children. In this way, we
#    have an one-on-one mapping between the plan tree and the DFS order sequence.
#    Notice2: Since the different plans have the different number of nodes, we need padding to make the lengths of the
#    features of different plans equal.

class PlanFeatureCollector:
    def __init__(self):
        # YOUR CODE HERE: define variables to collect features from plans
        self.features = []

    def add_operator(self, op: Operator):
        # YOUR CODE HERE: extract features from op
        est_rows = op.est_rows
        operator_type = op.id.split('_')[0]
        if op.is_table_scan():
            operator_enc = operators_enc_dict["TableScan"].copy()
        elif op.is_index_scan():
            operator_enc = operators_enc_dict["IndexScan"].copy()
        else:
            operator_enc = operators_enc_dict[operator_type].copy()
        operator_enc.append(float(est_rows))
        # self.features.append(operator_enc)
        self.features += operator_enc

    def walk_operator_tree(self, op: Operator):
        self.add_operator(op)
        for child in op.children:
            self.walk_operator_tree(child)
        # YOUR CODE HERE: process and return the features
        self.features += [0] * (len(operators) + 1)
        # self.features.append([0] * (len(operators) + 1))
        return self.features


class PlanDataset(torch.utils.data.Dataset):
    def __init__(self, plans, max_operator_num):
        super().__init__()
        self.data = []
        i = 0
        feature_length = (len(operators) + 1) * max_operator_num
        for plan in plans:
            i += 1

            collector = PlanFeatureCollector()
            vec = collector.walk_operator_tree(plan.root)
            # YOUR CODE HERE: maybe you need padding the features if you choose the second way to extract the features.
            if len(vec) < feature_length:
                difference = feature_length - len(vec)
                vec += [0] * difference
            features = torch.Tensor(vec)
            exec_time = torch.Tensor([plan.exec_time_in_ms()])
            self.data.append((features, exec_time))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


# Define your model for cost estimation
class YourModel(nn.Module):
    def __init__(self, input_layers):
        super(YourModel, self).__init__()
        self.fc1 = nn.Linear(input_layers, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = F.relu(self.fc4(out))
        return out

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                # nn.init.normal_(m.weight.data, 0, 0.1)
                nn.init.uniform_(m.weight.data, -0.0001, 0.0001)
                m.bias.data.zero_()
            elif isinstance(m, nn.LSTM):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()


def count_operator_num(op: Operator):
    num = 2  # one for the node and another for the end of children
    for child in op.children:
        num += count_operator_num(child)
    return num


def estimate_learning(train_plans, test_plans):
    max_operator_num = 0
    for plan in train_plans:
        max_operator_num = max(max_operator_num, count_operator_num(plan.root))
    for plan in test_plans:
        max_operator_num = max(max_operator_num, count_operator_num(plan.root))
    print(f"max_operator_num:{max_operator_num}")

    train_dataset = PlanDataset(train_plans, max_operator_num)
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=False, num_workers=1)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=False)

    input_layers = (len(operators) + 1) * max_operator_num

    model = YourModel(input_layers)
    model.init_weights()

    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # 设置学习率下降策略

    def loss_fn(est_time, act_time):
        # YOUR CODE HERE: define loss function
        return torch.mean(torch.abs(est_time - act_time) / act_time)

    # YOUR CODE HERE: complete training loop
    train_curve = []
    log_interval = 10
    num_epoch = 2000
    for epoch in range(num_epoch):
        print(f"epoch {epoch} start")
        model.train()
        loss_mean = 0.
        for i, data in enumerate(train_loader):
            inputs, labels = data
            # inputs = inputs.view(-1, 1, input_layers)
            outputs = model(inputs)

            # backward
            optimizer.zero_grad()
            loss = loss_fn(outputs, labels)
            loss.backward()

            # update weights
            optimizer.step()

            # print train information
            loss_mean += loss.item()
            # print(loss.item())
            train_curve.append(loss.item())
            if (i + 1) % log_interval == 0:
                loss_mean = loss_mean / log_interval
                print("Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f}".format(
                    epoch, num_epoch, i + 1, len(train_loader), loss_mean))
                loss_mean = 0.
        # scheduler.step()  # 更新学习率
    # 打印模型参数
    # for name in model.state_dict():
    #     print(name)
    #     print(model.state_dict()[name])

    # save model
    torch.save(model, './doc/lab2.pkl')

    train_est_times, train_act_times = [], []
    for i, data in enumerate(train_loader):
        # YOUR CODE HERE: evaluate on train data
        inputs, labels = data
        train_act_times += labels.squeeze().tolist()
        outputs = model(inputs)
        train_est_times += list(outputs.squeeze().cpu().detach().numpy())

    test_dataset = PlanDataset(test_plans, max_operator_num)
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=True, num_workers=1)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10)

    test_est_times, test_act_times = [], []
    for i, data in enumerate(test_loader):
        # YOUR CODE HERE: evaluate on test data
        inputs, labels = data
        test_act_times += labels.squeeze().tolist()
        outputs = model(inputs)
        test_est_times += list(outputs.squeeze().cpu().detach().numpy())
    test_est_times = [np.float64(i) for i in test_est_times]

    return train_est_times, train_act_times, test_est_times, test_act_times


if __name__ == '__main__':
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

    act_times = []
    tidb_costs = []
    for p in test_plans:
        act_times.append(p.exec_time_in_ms())
        tidb_costs.append(p.tidb_est_cost())

    print("done")

    _, _, est_learning_costs, act_learning_times = estimate_learning(train_plans, test_plans)

# print(est_learning_costs[:10])
# print(act_learning_times[:10])
