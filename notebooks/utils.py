import math
import numpy as np
import scipy
import pandas as pd


def pca(adata, n_components=50, train_ratio=0.35, n_batches=50, gpu=False):

    """
    Performs a batched PCA by training on the first `train_ratio` samples
    and transforming in `n_batches` number of batches.

    Parameters
    ----------

    adata : anndata.AnnData of shape (n_cells, n_genes)
        Annotated data object for which to perform PCA

    n_components : int
        Number of principal components to keep

    train_ratio : float
        Percentage of cells to use for training

    n_batches : int
        Number of batches to use for transform

    gpu : bool
        Uses Scikit-Learn for CPU (gpu=False) and RAPIDS cuML for GPU
        (gpu=True)
    """

    train_size = math.ceil(adata.X.shape[0] * train_ratio)

    if gpu:
        from cuml.decomposition import PCA
        import cupy as cp
    else:
        from sklearn.decomposition import PCA
        import numpy as cp

    pca = PCA(n_components=n_components).fit(adata.X[:train_size])

    embeddings = cp.zeros((adata.X.shape[0], n_components))
    batch_size = int(embeddings.shape[0] / n_batches)
    for batch in range(n_batches):
        start_idx = batch * batch_size
        end_idx = start_idx + batch_size

        if(adata.X.shape[0] - end_idx < batch_size):
            end_idx = adata.X.shape[0]

        embeddings[start_idx:end_idx,:] = cp.asarray(pca.transform(adata.X[start_idx:end_idx]))

    if gpu:
        embeddings = embeddings.get()

    adata.obsm["X_pca"] = embeddings
    return adata


def tf_idf(filtered_cells):
    '''
    Input: 2D numpy.ndarray or 2D sparse matrix with X[i, j] = (binary or continuous) read count for cell i, peak j
    Output: Normalized matrix, where Xp[i, j] = X[i, j] * (1 / sum_peaks(i)) * log(1 + N_cells/N_cells with peak j)
    Note that the 1 / sum_peaks(i) term isn't included in the standard NLP form of tf-idf, but other single-cell work uses it.
    '''
    inv_sums = 1 / np.array(filtered_cells.sum(axis=1)).ravel()

    peak_counts = np.array((filtered_cells > 0).sum(axis=0)).ravel()
    log_inv_peak_freq = np.log1p(filtered_cells.shape[0] / peak_counts)

    normalized = filtered_cells.multiply(inv_sums[:, np.newaxis])
    normalized = normalized.multiply(log_inv_peak_freq[np.newaxis, :])
    normalized = scipy.sparse.csr_matrix(normalized)

    return normalized


def logtf_idf(filtered_cells, pseudocount=10**5):
    '''
    Input: 2D numpy.ndarray or 2D sparse matrix with X[i, j] = (binary or continuous) read count for cell i, peak j
    Output: Normalized matrix, where Xp[i, j] = X[i, j] * log(1 + pseudocount/sum_peaks(i)) * log(1 + N_cells/N_cells with peak j)
    Pseudocount should be chosen as a similar order of magnitude as the mean number of reads per cell.
    '''
    log_inv_sums = np.log1p(pseudocount / np.array(filtered_cells.sum(axis=1)).ravel())

    peak_counts = np.array((filtered_cells > 0).sum(axis=0)).ravel()
    log_inv_peak_freq = np.log1p(filtered_cells.shape[0] / peak_counts)

    normalized = filtered_cells.multiply(log_inv_sums[:, np.newaxis])
    normalized = normalized.multiply(log_inv_peak_freq[np.newaxis, :])
    normalized = scipy.sparse.csr_matrix(normalized)

    return normalized


def overlap(gene, fragment, upstream=10000, downstream=0):
    '''
    Checks if a genomic interval ('fragment') overlaps a gene, or some number of bases upstream/downstream
    of that gene.
    '''
    if gene[3] == 'rev':
        t = upstream
        upstream = downstream
        downstream = t
    if gene[0] != fragment[0]: # check chromosome
        return False
    if gene[2] + downstream >= fragment[1] and gene[1] - upstream <= fragment[1]: # peak start in gene
        return True
    if gene[2] + downstream >= fragment[2] and gene[1] - upstream <= fragment[2]: # peak end in gene
        return True
    if gene[1] - upstream >= fragment[1] and gene[2] + downstream <= fragment[2]: # gene entirely within peak
        return True


def filter_peaks(adata, n_top_peaks):
    '''
    Retains the top N most frequent peaks in the count matrix.
    '''
    peak_occurrences = np.sum(adata.X > 0, axis=0)
    peak_frequency = np.array(peak_occurrences / adata.X.shape[0]).flatten()
    frequent_peak_idxs = np.argsort(peak_frequency)
    use = frequent_peak_idxs[-n_top_peaks : ]
    return adata[:, use]

