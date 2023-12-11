import pandas as pd
import lz4.frame
import gzip
import zstandard as zstd
import io
import json


def get_compression_sizes(df):

    buffer = io.BytesIO()
    df.to_pickle(buffer)
    buffer.seek(0)
    df_pkl = buffer.getvalue()
    original_size = len(df_pkl)

    lz4_compressed = lz4.frame.compress(df_pkl)
    lz4_size = len(lz4_compressed)
    original_size = len(df_pkl)

    gzip_buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=gzip_buffer, mode='wb') as f:
        f.write(df_pkl)
    gzip_size = len(gzip_buffer.getvalue())

    zstd_compressor = zstd.ZstdCompressor()
    zstd_compressed = zstd_compressor.compress(df_pkl)
    zstd_size = len(zstd_compressed)

    return {'original_size': original_size, 'lz4': lz4_size, 'gzip': gzip_size, 'zstandard': zstd_size}


def compress_with_FD_linear(df, FD_list, save_file_name):
    original_compression = get_compression_sizes(df)
    sorted_column_compression = get_compression_sizes(df.sort_values(by=FD_list))
    fully_sorted_compression = get_compression_sizes(df.sort_values(by=df.columns.tolist()))

    results = {
        'original': original_compression,
        'sorted_by_columns': sorted_column_compression,
        'fully_sorted': fully_sorted_compression
    }

    # Save results to a JSON file
    with open(save_file_name, 'w') as f:
        json.dump(results, f, indent=4)

    return results

