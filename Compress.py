import io
import gzip
import lz4.frame as lz4
import zstandard as zstd
import pandas as pd

Result_path_surfix = "Result/"

def random_cluster_dataframe(df, n_clusters, seed):
    if seed is not None:
        np.random.seed(seed)
    df['cluster'] = np.random.choice(n_clusters, len(df))
    return df

def get_compressed_size_Gzip(column_data):
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode='w') as f:
        f.write(column_data.encode('utf-8'))
    compressed_size = buffer.tell()
    buffer.close()
    return compressed_size

# Compression method for LZ4
def get_compressed_size_LZ4(column_data):
    compressed_data = lz4.compress(column_data.encode('utf-8'))
    return len(compressed_data)

# Compression method for Zstandard
def get_compressed_size_Zstandard(column_data):
    cctx = zstd.ZstdCompressor()
    compressed_data = cctx.compress(column_data.encode('utf-8'))
    return len(compressed_data)


def total_compression_size(df, compression_method):
    total_compressed_size = 0
    compression_func = None

    if compression_method == 'gzip':
        compression_func = get_compressed_size_Gzip
    elif compression_method == 'lz4':
        compression_func = get_compressed_size_LZ4
    elif compression_method == 'zstandard':
        compression_func = get_compressed_size_Zstandard
    else:
        raise ValueError(f"Unknown compression method: {compression_method}")

    # Data to save to Excel
    results = []

    for column in df.columns:
        column_data = df[column].to_string(index=False)
        compressed_size = compression_func(column_data)
        total_compressed_size += compressed_size

        # Append result to the list
        results.append({'Column': column, 'Compressed Size (bytes)': compressed_size})

    results.append({'Column': 'Total', 'Compressed Size (bytes)': total_compressed_size})

    # Convert results to a DataFrame and return
    results_df = pd.DataFrame(results)
    return results_df


def compress_clusters(df, n_clusters, compression_method, seed):
    # Randomly cluster the dataframe
    df_clustered = random_cluster_dataframe(df, n_clusters, seed)

    # Data to save to Excel for total compression across clusters
    total_results = []

    grand_total_compressed_size = 0

    # Create an Excel writer object
    with pd.ExcelWriter('compression_results_all_clusters.xlsx') as writer:
        for cluster in range(n_clusters):
            cluster_df = df_clustered[df_clustered['cluster'] == cluster].drop('cluster', axis=1)
            results_df = total_compression_size(cluster_df, compression_method)
            results_df.to_excel(writer, sheet_name='Cluster_{}'.format(cluster), index=False)

            cluster_total_size = results_df[results_df['Column'] == 'Total']['Compressed Size (bytes)'].iloc[0]
            grand_total_compressed_size += cluster_total_size
            total_results.append({'Cluster': cluster, 'Compressed Size (bytes)': cluster_total_size})

        total_results.append({'Cluster': 'Total', 'Compressed Size (bytes)': grand_total_compressed_size})
        # Convert total_results to a DataFrame and add to the same Excel file
        total_results_df = pd.DataFrame(total_results)
        total_results_df.to_excel(writer, sheet_name=df.name + "_" + str(n_clusters) + "_" + compression_method, index=False)


def compress_clusters_summary(df, n_clusters, compression_method, seed):
    # Randomly cluster the dataframe
    df_clustered = random_cluster_dataframe(df, n_clusters, seed)

    # Create a DataFrame to hold aggregate results
    aggregate_df = pd.DataFrame()

    # Use df.columns.tolist() instead of list(df.columns)
    columns = df.columns.tolist()
    columns.append('Total')
    aggregate_df['Column'] = columns

    # Iterate over clusters, get their compression sizes, and add to aggregate_df
    for cluster in range(n_clusters):
        cluster_df = df_clustered[df_clustered['cluster'] == cluster].drop('cluster', axis=1)
        results_df = total_compression_size(cluster_df, compression_method)

        aggregate_df[f'Cluster_{cluster}'] = results_df['Compressed Size (bytes)']

    # Sum up all clusters for a grand total
    aggregate_df['Total Compression'] = aggregate_df.iloc[:, 1:].sum(axis=1)

    # Save aggregate_df to Excel
    aggregate_df.to_excel(df.name + "_" + str(n_clusters) + "_" + compression_method + ".xlsx", index=False)

