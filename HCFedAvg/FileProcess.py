
import pandas as pd

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

    # create_file("")
    # add_new_column("Config_name")
    # add_new_column_with_file("L2", "TAS_result")

    # add_new_column('sim_std')
    # args = Args.Arguments()
    # res_1 = read_row(48)
    # # 将字符串转换为Python列表
    # float_list = ast.literal_eval(res_1['acc_list'])
    #
    # # 使用列表推导式将字符串列表转换为浮点数列表
    # float_list = [float(x) for x in float_list]
    # # print(max(float_list[:200]))
    # # print(float_list)
    # # print(float_list[72])
    # count = 0
    # had = [False, False, False]
    # for i, acc in enumerate(float_list):
    #     if count == 0 and acc >=0.75:
    #         count +=1
    #         continue
    #     if acc >= 0.75 and had[0] is False:
    #         had[0] = True
    #         print(i)
    #     if acc >= 0.77 and had[1] is False:
    #         had[1] = True
    #
    #         print(i)
    #     if acc >= 0.79 and had[2] is False:
    #         had[2] = True
    #         print(i)
    pass



