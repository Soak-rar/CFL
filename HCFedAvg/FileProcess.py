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


def add_row_with_file_name(result_dict, file_name):
    df = pd.read_csv(file_name + '.csv')

    df = df._append(result_dict, ignore_index=True)
    # 保存更新后的DataFrame回表格文件
    df.to_csv(file_name + '.csv', index=False)


def create_file(filename, save_dict):

    header = save_dict.keys()

    # 创建DataFrame只包含表头行
    df = pd.DataFrame(columns=header)

    # 将DataFrame保存为CSV文件
    df.to_csv(filename + '.csv', index=False)


def create_file_with_header(filename, header):

    # 创建DataFrame只包含表头行
    df = pd.DataFrame(columns=header)

    # 将DataFrame保存为CSV文件
    df.to_csv(filename + '.csv', index=False)


def add_new_column(column_name):
    df = pd.read_csv('result.csv')
    df[column_name] = None
    df.to_csv('result.csv', index=False)

def add_new_column_with_file(column_name, file_name):
    df = pd.read_csv(file_name + '.csv')
    df[column_name] = None
    df.to_csv(file_name + '.csv', index=False)

def read_all_rows():
    pass


def read_row(row_id):
    df = pd.read_csv('result.csv')

    # 选择第n行，行号从0开始计数
    selected_row = df.iloc[row_id]
    return selected_row


def read_row_with_file_name(row_id, filename):
    df = pd.read_csv(filename + '.csv')

    # 选择第n行，行号从0开始计数
    selected_row = df.iloc[row_id]
    return selected_row

if __name__ == '__main__':

    create_file("")
    add_new_column("extra_param")

    # add_new_column('sim_std')
    # args = Args.Arguments()
    # res_1 = read_row(35)
    # # 将字符串转换为Python列表
    # float_list_1 = ast.literal_eval(res_1['sim_mean'])
    #
    # res_2 = read_row(36)
    # float_list_2 = ast.literal_eval(res_2['sim_mean'])
    # for i in range(len(float_list_1)):
    #
    #     print(float_list_1[i] - float_list_2[i])
    # print(res['acc'])
    # # 使用列表推导式将字符串列表转换为浮点数列表
    # float_list = [float(x) for x in float_list]
    # # print(max(float_list[:200]))
    # # print(float_list)
    # print(res['algorithm_name'])
    # had = [False, False, False]
    # for i, acc in enumerate(float_list):
    #     if acc >= 0.80 and had[0] is False:
    #         had[0] = True
    #         print(i)
    #     if acc >= 0.82 and had[1] is False:
    #         had[1] = True
    #         print(i)
    #     if acc >= 0.84 and had[2] is False:
    #         had[2] = True
    #         print(i)
    pass



