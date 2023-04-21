from plan import Plan
import json

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

# _, _, est_learning_costs, act_learning_times = estimate_learning(train_plans, test_plans)

# print(est_learning_costs[:10])
# print(act_learning_times[:10])