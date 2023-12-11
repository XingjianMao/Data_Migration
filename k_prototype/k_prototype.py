import json
import sys
import pandas as pd
from datetime import datetime
from kmodes.kprototypes import KPrototypes
from sklearn.preprocessing import LabelEncoder
import time

#先把不能处理的长string的column删了，把nan填上0，再把处理后的cat col的新indice弄出来
def preprocess_dataframe(df, remove_indice, cat_col):

    df_dropped = df.drop(df.columns[remove_indice], axis=1)

    adjusted_cat_col = []
    for index in cat_col:
        adjusted_index = index - sum(i < index for i in remove_indice)
        adjusted_cat_col.append(adjusted_index)
    df_dropped = df_dropped.fillna("0")
    return df_dropped, adjusted_cat_col


#取一下config
def Get_config_json_kproto(config_file):
    with open(config_file, 'r') as file:
        config = json.load(file)

    #数据名字
    data_name = config["data_name"]
    #数据的path
    data_path = config['data_path']
    #多少个cluster
    cluster_size = config['cluster_size']

    #然后是一下k proto的参数
    n_jobs = config['n_jobs']
    n_init = config['n_init']
    max_iter = config['max_iter']
    
    #这个是切割符
    delimiter = config["delimiter"]

    #选择要drop掉的col，就是无法处理的
    list_drop = config["column_drop"]
    #选择可以处理的cat col
    list_cat = config["column_cat_col"]


    return data_name, data_path, delimiter, cluster_size, n_jobs, n_init, max_iter, list_drop, list_cat



def k_prototype(config_path, clustered_save_path):
    #读一下data，读一下config，设置kprototype
    data_name, data_path, delimiter, cluster_size, n_jobs, n_init, max_iter, list_drop, list_cat = Get_config_json_kproto(config_path)
    df = pd.read_csv(data_path, header = 0, delimiter= delimiter)
    kp = KPrototypes(n_clusters=cluster_size, n_jobs=n_jobs, max_iter=max_iter, n_init=n_init)

    start_time = time.time()
    #preprocess一下，new list cat是cat col，因为预处理的时候要drop掉一些col，所以要重新弄一下
    df_pre_process, new_list_cat = preprocess_dataframe(df, list_drop, list_cat)
    #predict
    clusters = kp.fit_predict(df_pre_process, categorical=new_list_cat)
    end_time = time.time()

    #算一下时间
    run_time = end_time - start_time
    print("Time to done the clustering: " + str(run_time))

    df['cluster_id'] = clusters

    #把填上cluster id的csv保存到/Dataset/k_prototype_clustere文件夹
    file_name = clustered_save_path + "/k_proto_" + str(cluster_size) + "_" + data_name + ".csv"
    df.to_csv(file_name, index=False)

    
if __name__ == '__main__':
    default_config_path = "config_folder/config_flight.json"
    default_save_path = "Dataset/k_proto_clustered"
    k_prototype(default_config_path, default_save_path)




