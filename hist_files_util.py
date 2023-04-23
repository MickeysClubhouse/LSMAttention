import pandas as pd

df = pd.read_csv('data/histogram/hist_kind_id.csv', delimiter='|')
df = df.iloc[:, [4, 7, 9, 10]]

new_columns = []
for col in df.columns:
    new_columns.append(col.replace(" ", ""))

df.columns = new_columns

# 计算当前count值和上一行的count值之差
df['Count_diff'] = df['Count'] - df['Count'].shift(1)

# 将第一行的count_diff值设为NaN
df.loc[0, 'Count_diff'] = df['Count'][0]

print(df)
