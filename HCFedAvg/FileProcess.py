import os
import pandas as pd
import Args
import ast
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


def read_all_rows():
    pass


def read_row(row_id):
    df = pd.read_csv('result.csv')

    # 选择第n行，行号从0开始计数
    selected_row = df.iloc[row_id]
    return selected_row

if __name__ == '__main__':
    args = Args.Arguments()
    res = read_row(2)
    # 将字符串转换为Python列表
    float_list = ast.literal_eval(res['acc_list'])
    # 使用列表推导式将字符串列表转换为浮点数列表
    float_list = [float(x) for x in float_list]
    # print(max(float_list[:200]))
    # print(float_list)
    print(res['algorithm_name'])
    had = [False, False, False]
    for i, acc in enumerate(float_list):
        if acc >= 0.95 and had[0] is False:
            had[0] = True
            print(i)
        if acc >= 0.97 and had[1] is False:
            had[1] = True
            print(i)
        if acc >= 0.99 and had[2] is False:
            had[2] = True
            print(i)



