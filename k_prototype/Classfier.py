import json
import pandas as pd
import time
import os
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

def Get_config_json_Classifier(config_file):
    with open(config_file, 'r') as file:
        config = json.load(file)
    
    data_name = config["data_name"]
    cat_col = config["column_cat_col"]
    cluster_size = config['cluster_size']
    col_drop = config["column_drop"]
    delimiter = config["delimiter"]
    test_ratio = config["test_ratio"]

    return data_name, cluster_size, cat_col, col_drop, delimiter, test_ratio

def Preprocess_Classifier(df, cat_col, drop_col):
    le = LabelEncoder()
    for index in cat_col:
        if index < len(df.columns):
            column_name = df.columns[index]
            df[column_name] = le.fit_transform(df[column_name])
        else:
            print(f"Index {index} is out of bounds for the DataFrame.")
    
    df = df.drop(df.columns[drop_col], axis=1)
    df = df.fillna("0")
    return df

def run_main():

    config_path = "config_folder/config_nypd.json"
    data_name, cluster_size, cat_col, col_drop, delimiter, train_ratio = Get_config_json_Classifier(config_path)

    data_path = "Dataset/k_proto_clustered/k_proto_10_nypd.csv"
    df = pd.read_csv(data_path, header=0, delimiter=delimiter)
    df_processed = Preprocess_Classifier(df, cat_col, col_drop)

    X = df_processed.drop('cluster_id', axis=1)  
    y = df_processed['cluster_id']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=train_ratio, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

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

    current_time = time.time()
    folder_path = 'Classifier_result/' + data_name + "_" + str(cluster_size) + "_" + str(current_time)
    os.makedirs(folder_path, exist_ok=True)

    training_times = {}

    for name, clf in classifiers.items():
        start_time = time.time()
        clf.fit(X_train_scaled, y_train)
        end_time = time.time()

        training_time = end_time - start_time
        print("Train time for " + name + " is " + str(training_time))
        training_times[name] = training_time

        filename = folder_path + "/" + name + "_" + str(train_ratio) + "_classifier.joblib"
        dump(clf, filename)
        print(f'Saved {name} classifier to {filename}')

    with open(folder_path + "/training_times.json", 'w') as file:
        json.dump(training_times, file, indent=4)
    print("Saved training times to JSON file.")

    predictions_df = df.copy()

    for name, clf in classifiers.items():
        predictions = clf.predict(scaler.transform(X))
        predictions_df[name + '_Cluster_Prediction'] = predictions

    predictions_csv_path = folder_path + "/all_predictions_with_original_data.csv"
    predictions_df.to_csv(predictions_csv_path, index=False)
    print(f"Saved all predictions with original data to {predictions_csv_path}")


if __name__ == '__main__':
    run_main()
