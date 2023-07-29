import os
import pandas as pd
import Args

# 实验结果csv文件处理

def add_row(result_dict):
    df = pd.read_csv('result.csv')

    df = df._append(result_dict, ignore_index=True)
    # 保存更新后的DataFrame回表格文件
    df.to_csv('result.csv', index=False)


def create_file(filename, args: Args.Arguments):

    save_row = args.save_dict()
    header = save_row.keys()

    # 创建DataFrame只包含表头行
    df = pd.DataFrame(columns=header)

    # 将DataFrame保存为CSV文件
    df.to_csv(filename + '.csv', index=False)


if __name__ == '__main__':
    pass
    # args = Args.Arguments()
    # create_file('result', args)
    # add_row(args.save_dict())


