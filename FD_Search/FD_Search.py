import time
import pandas as pd
import gzip
import io
from scipy.stats import entropy
import math
import itertools
from collections import defaultdict
from itertools import combinations
import numpy as np
import lz4.frame as lz4
import zstandard as zstd
import networkx as nx
from math import sqrt
import json
from collections import Counter
from FD_Mutual_Info import *
import argparse
from datetime import datetime
import os
from Test_compress_linear import *
from Test_compress_column import *

def remove_mismatched_types(df, expected_types):
    # Iterate over each row
    for index, row in df.iterrows():
        # Check each cell in the row
        for col, expected_type in zip(row, expected_types):
            if expected_type == 'int' and not pd.api.types.is_integer(col):
                # Drop the row if the type is not integer
                df.drop(index, inplace=True)
                break
            elif expected_type == 'str' and not isinstance(col, str):
                # Drop the row if the type is not string
                df.drop(index, inplace=True)
                break
    return df

#这个就是读一下data，然后取差不多一万个sample来算FD
def Get_data (data_path, sample_num, random_seed, delimiter):

    df_original = pd.read_csv(data_path, header = 0, delimiter = delimiter)
    df = df_original.sample(n = sample_num, random_state=random_seed)
    df_sample = df.reset_index(drop=True)
    df = df.reset_index(drop=True)
    
    return df_original, df_sample

#这个是读取config文件
def Get_config_json_FD(config_file):
    with open(config_file, 'r') as file:
        config = json.load(file)
    #这个是原来文件的地址
    data_path = config['data_path']
    #这个是sample的多少，我设置的是一万
    sample_num = config['sample_num']
    #这个是取sample的种子，随便设置一个数字，我的好像是67
    sample_seed = config['sample_seed']
    #这个是不变的，算fd要用的一个参数
    eps_3 = config['epsilon_3']
    #这个也是一个不变的参数
    eps_2 = config['epsilon_2']
    #这个也是
    prune_threshold = config['prune_threshold']

    #这个是用来命名记录文件的，一半如果原数据集叫NYPD.CSV，这个就叫nypd
    data_name = config["data_name"]
    #这个是原数据切割符
    delimiter = config["delimiter"]

    return data_path, delimiter, sample_num, sample_seed, eps_3, eps_2, prune_threshold, data_name


def Find_FD(config_file_path):
    #config file path 就是我这个config 的位置

    #parameter从config读出来
    data_path, delimiter, sample_num, sample_seed, eps_3, eps_2, prune_threshold, data_name = Get_config_json_FD(config_file_path)

    #读下数据，塞进一个dataframe
    df_original, df_sample = Get_data(data_path, sample_num, sample_seed, delimiter)


    start_time = time.time()
    #先裁剪一下数据，把这个不变的参数放进去就行
    prune_df = prune_data(df_sample, eps_3, prune_threshold)
    #Entropy_cal就是算FD的方法，出来的FD_LIST就是记录FD relation的list，看上去是这样的[("Col_A", "Col_B"), ("Col_D", "Col_E")]
    FD_list = Entropy_Cal(prune_df, eps_2)
    end_time = time.time()
    #算一下运行时间
    time_taken = end_time - start_time
    #画个图
    current_time = time.time()
    folder_path = 'FD_result/' + data_name + str(current_time)
    os.makedirs(folder_path, exist_ok=True)

    G = nx.DiGraph()
    G.add_edges_from(FD_list)

    plot_graph(G, data_name, folder_path)

    #这个不用管，就是算一下FD里面出现最多的column是哪个
    flattened_data = [item for sublist in FD_list for item in sublist]
    counter = Counter(flattened_data)

    # 排序一下，这个也不用管
    sorted_items = sorted(counter.items(), key=lambda x: x[1], reverse=True)

    #算下这个list里面每个column name的数量，排序一下，然后按这个顺序排列
    print(sorted_items)
    sorted_list_column = [item[0] for item in sorted_items]
    print(sorted_list_column)
    

    #selected就是在这个FD网中取几个重要的节点，这几个节点就是用来sort整个表格的
    
    selected = select_nodes_OneEachSubgraph(FD_list)
    #最后把结果放到一个json里面

    file_path_compress = folder_path + "/" + data_name + "_CompressResult.json"


    compress_with_FD_linear(df_original, selected, file_path_compress)
    Experiment_compress_column(df_original, selected, folder_path)
    #column_experiment(df_original, folder_path)

    lists_dict = {
        "time_taken": time_taken,
        'sorted_items': sorted_items,
        'sorted_list_column': sorted_list_column,
        'FD_list': FD_list,
        "Selected" : selected
    }


    file_path_FD = folder_path + "/" + data_name + "_FD.json"
    with open(file_path_FD, 'w') as file:
        json.dump(lists_dict, file)

    


if __name__ == '__main__':
    
    default_file_path = "config_folder/config_ncvoter_million.json"
    
    parser = argparse.ArgumentParser(description='the config file path of k prototype clustering')
    parser.add_argument('--config_path', type=str, default=default_file_path, help='Input string')
    args = parser.parse_args()

    config_path = args.config_path
    
    Find_FD(config_path)





