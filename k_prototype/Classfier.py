import json
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import time
from joblib import dump
import os

#弄一下config
def Get_config_json_Classifier(config_file):
    with open(config_file, 'r') as file:
        config = json.load(file)
    
    #cat col就是可以处理的catogorical column
    cat_col = config["column_cat_col"]

    #多少个cluster
    cluster_size = config['cluster_size']
    #要drop掉的不能处理的col
    col_drop = config["column_drop"]
    delimiter = config["delimiter"]

    return cluster_size, cat_col, col_drop, delimiter

#预处理和k prototype不太一样，因为可以处理的cat col要label encoding，drop掉不能处理的列，然后nan填上0
def Preprocess_Classifier(df, cat_col, drop_col):
    le = LabelEncoder()
    for index in cat_col:
        # Ensure the index is within the range of DataFrame columns
        if index < len(df.columns):
            column_name = df.columns[index]
            df[column_name] = le.fit_transform(df[column_name])
        else:
            print(f"Index {index} is out of bounds for the DataFrame.")
    
    df = df.drop(df.columns[drop_col], axis=1)
    df = df.fillna("0")
    return df





def run_main():

    #读一下config
    config_path = "config_folder/config_nypd.json"
    cluster_size, cat_col, col_drop, delimiter = Get_config_json_Classifier(config_path)

    #这里的path是已经k prototype好的数据
    data_path = "Dataset/k_proto_clustered/k_proto_5_nypd.csv"
    
    df = pd.read_csv(data_path, header = 0, delimiter= delimiter)

    #与处理下
    df_Processed= Preprocess_Classifier(    df, cat_col, col_drop)

    #
    X = df_Processed.drop('cluster_id', axis=1)  
    y = df_Processed['cluster_id']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    classifiers = {
        'DecisionTree': DecisionTreeClassifier(),
        'RandomForest': RandomForestClassifier(n_estimators=10, n_jobs=-1),
        'AdaBoost': AdaBoostClassifier(n_estimators=10, estimator=DecisionTreeClassifier(max_depth=10)),
        'QDA': QuadraticDiscriminantAnalysis(),
        'MLP': MLPClassifier(hidden_layer_sizes=(10,)),
        'GaussianNB': GaussianNB(),
        'KNN': KNeighborsClassifier(n_jobs=-1),
        'LogisticRegression': LogisticRegression(n_jobs=-1)
    }

    #训练，训练好了放到Save_Classfier文件夹里面
    for name, clf in classifiers.items():
        start_time = time.time()
        clf.fit(X_scaled, y)
        end_time = time.time()

        training_time = end_time - start_time
        print("Train time for " + name + " is " + str(training_time))
        filename = "Save_Classifier/" + name + "_classifier.joblib"

        dump(clf, filename)
        print(f'Saved {name} classifier to {filename}')


if __name__ == '__main__':
    run_main()
    
