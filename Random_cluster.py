
import numpy as np


def random_cluster_dataframe(df, n_clusters, seed):
    if seed is not None:
        np.random.seed(seed)
    df['cluster'] = np.random.choice(n_clusters, len(df))
    return df