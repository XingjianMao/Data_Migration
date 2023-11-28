import pandas as pd
import numpy as np
import gzip
import zstandard as zstd
import lz4.frame
from io import BytesIO

def compress_column(column_data, method):
    if method == 'gzip':
        # Using gzip compression
        buffer = BytesIO()
        with gzip.GzipFile(fileobj=buffer, mode='wb') as f:
            f.write(column_data)
            compressed_data = buffer.getvalue()  # Moved inside the 'with' block
    elif method == 'lz4':
        # Using lz4 compression
        compressed_data = lz4.frame.compress(column_data)
    elif method == 'zstandard':
        # Using zstandard compression
        buffer = BytesIO()
        with zstd.ZstdCompressor().stream_writer(buffer) as compressor:
            compressor.write(column_data)
            compressed_data = buffer.getvalue()  # Moved inside the 'with' block
    else:
        raise ValueError(f"Unknown compression method: {method}")
    return compressed_data


def compress_and_calculate_sizes(df, sort_columns=None, method='gzip', sort_all=False):
    if sort_columns and not sort_all:
        df = df.sort_values(by=sort_columns)
    elif sort_all:
        df = df.sort_values(by=list(df.columns))
    compressed_sizes = []
    for column in df.columns:
        data = df[column].astype(str).str.cat(sep=' ').encode('utf-8')
        compressed_data = compress_column(data, method=method)
        compressed_sizes.append(len(compressed_data))
    return compressed_sizes

def process_clusters(data, cluster_id, column_names):
    results = {}
    compression_methods = ['gzip', 'zstandard', 'lz4']
    sorting_options = [('sort_all', True, None), 
                       ('sort_based_on_FD', False, column_names), 
                       ('no_sorting', False, None)]
    
    clusters = data[cluster_id].unique()
    
    for method in compression_methods:
        for sort_label, sort_all, sort_columns in sorting_options:
            key = f'({method}) {sort_label}'
            results[key] = []
            for cluster in clusters:
                cluster_data = data[data[cluster_id] == cluster]
                sizes = compress_and_calculate_sizes(cluster_data, sort_columns, method, sort_all)
                results[key].append(sizes)

    final_results = {key: [sum(x) for x in zip(*value)] for key, value in results.items()}
    final_results['column'] = [f'column_{i+1}' for i in range(len(sizes))] + ['Total']
    
    for key in results:
        final_results[key].append(sum(final_results[key]))
    
    return pd.DataFrame(final_results)


data = pd.read_csv('Classifier_result/Clustered_data/modified_RandomForest.csv') 
column_names = ["capital-gain", "hours-per-week"] 


result_df = process_clusters(data, 'cluster_id', column_names)


output_filename = 'Test_Compress/Compress_result/RandomForest.csv'
result_df.to_csv(output_filename, index=False)
output_filename
