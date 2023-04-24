import pandas as pd
import os


def rebin(df: pd.DataFrame):
    bins = []
    intervals = []
    bin_num = 0
    for idx, row in df.iterrows():
        bin_num += 1
        bins.append(row['Lower_Bound'])
        if bin_num > 1:
            intervals.append(bins[-1] - bins[-2])

        # 左闭右开
    target_num = 50
    ratio = (bin_num - 1) / target_num

    # build new bins
    new_bins = [bins[0]]  # target bins
    base = bins[0]  # base number,累加的
    int_num = 0  # interval of old bin
    for i in range(50):
        tmp = base + intervals[int_num] * ratio

        # 边界条件
        if i == 49:
            new_bins.append(bins[-1])
            break
        if tmp > bins[int_num + 1]:  # 越过右边
            remain = ratio - (bins[int_num + 1] - base) / intervals[int_num]
            # 移动到下一个区域
            int_num += 1
            base = bins[int_num]
            tmp = base + remain * intervals[int_num]

        new_bins.append(tmp)
        base = tmp

    return new_bins


def get_bins(filename):
    df = pd.read_csv(filename, delimiter='|')
    df = df.iloc[:, [4, 7, 9, 10]]  # 需要的属性

    new_columns = []
    for col in df.columns:
        new_columns.append(col.replace(" ", ""))

    df.columns = new_columns

    # 计算当前count值和上一行的count值之差
    df['Count_diff'] = df['Count'] - df['Count'].shift(1)

    # 将第一行的count_diff值设为count
    df.loc[0, 'Count_diff'] = df['Count'][0]

    bins = rebin(df)
    return bins


def get_histograms():
    hists = []
    # 遍历文件夹
    for root, dirs, files in os.walk("data/histogram"):
        for filename in files:
            # 处理文件
            hists.append(get_bins(os.path.join(root, filename)))

    return hists


if __name__ == '__main__':
    histograms = get_histograms()
    print("asdasdasd")
