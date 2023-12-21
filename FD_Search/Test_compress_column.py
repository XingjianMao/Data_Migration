import io
import gzip
import lz4.frame as lz4
import zstandard as zstd
import pandas as pd

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

# Outputs the entire df's compress size. Compression method is the compression algorithm.
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

    # Convert results to a DataFrame and save to Excel
    results_df = pd.DataFrame(results)
    return results_df



def Experiment_compress_column(df, FD_list, save_file_path):
    result_original_lz4 = total_compression_size(df, "lz4")
    result_original_gzip = total_compression_size(df, "gzip")
    result_original_zstandard = total_compression_size(df, "zstandard")
    
    result_original_lz4.to_excel(save_file_path + "/original_lz4.xlsx", index=False)
    result_original_gzip.to_excel(save_file_path + "/original_gzip.xlsx", index=False)
    result_original_zstandard.to_excel(save_file_path + "/original_zstandard.xlsx", index=False)

    result_FD_lz4 = total_compression_size(df.sort_values(by=FD_list), "lz4")
    result_FD_gzip = total_compression_size(df.sort_values(by=FD_list), "gzip")
    result_FD_zstandard = total_compression_size(df.sort_values(by=FD_list), "zstandard")

    result_FD_lz4.to_excel(save_file_path + "/FD_sort_lz4.xlsx", index=False)
    result_FD_gzip.to_excel(save_file_path + "/FD_sort_gzip.xlsx", index=False)
    result_FD_zstandard.to_excel(save_file_path + "/FD_sort_zstandard.xlsx", index=False)

    result_FD_sortall_lz4 = total_compression_size(df.sort_values(by=df.columns.tolist()), "lz4")
    result_FD_sortall_gzip = total_compression_size(df.sort_values(by=df.columns.tolist()), "gzip")
    result_FD_sortall_zstandard = total_compression_size(df.sort_values(by=df.columns.tolist()), "zstandard")

    result_FD_sortall_lz4.to_excel(save_file_path + "/sortall_lz4.xlsx", index=False)
    result_FD_sortall_gzip.to_excel(save_file_path + "/sortall_gzip.xlsx", index=False)
    result_FD_sortall_zstandard.to_excel(save_file_path + "/sortall_zstandard.xlsx", index=False)


def column_experiment(df, path, filename="compression_results.xlsx"):
    compression_methods = {
        'gzip': get_compressed_size_Gzip,
        'lz4': get_compressed_size_LZ4,
        'zstandard': get_compressed_size_Zstandard
    }

    results = []
    
    for method_name, method_func in compression_methods.items():
        total_compressed_size = 0
        method_results = []

        for column in df.columns:
            sorted_df = df.sort_values(by=column)
            column_data = sorted_df[column].to_string(index=False)
            compressed_size = method_func(column_data)
            total_compressed_size += compressed_size
            method_results.append({'Column': column, 'Compressed Size (bytes)': compressed_size})

        method_results.append({'Column': 'Total', f'{method_name} Compressed Size (bytes)': total_compressed_size})
        results_df = pd.DataFrame(method_results)
        results.append(results_df)

    # Combine results into a single DataFrame
    final_df = pd.concat(results, axis=1)
    
    # Save to Excel
    final_df.to_excel(path + "/" + filename, index=False)
    print(f"Saved compression results to {filename}")
