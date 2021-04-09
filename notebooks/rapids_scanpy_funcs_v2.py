#
# Copyright (c) 2020, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import cupy as cp
import cudf
import cugraph
import anndata
import time

import numpy as np
import pandas as pd
import scipy
import math
from scipy import sparse
import seaborn as sns
import matplotlib.pyplot as plt


from cuml.manifold import TSNE
from cuml.cluster import KMeans
from cuml.decomposition import PCA
from cuml.linear_model import LinearRegression



    
def scale(normalized, max_value=10):
    """
    Scales matrix to unit variance and clips values

    Parameters
    ----------

    normalized : cupy.ndarray or numpy.ndarray of shape (n_cells, n_genes)
                 Matrix to scale
    max_value : int
                After scaling matrix to unit variance,
                values will be clipped to this number
                of std deviations.

    Return
    ------

    normalized : cupy.ndarray of shape (n_cells, n_genes)
        Dense normalized matrix
    """

    normalized = cp.asarray(normalized)
    mean = normalized.mean(axis=0)
    normalized -= mean
    del mean
    stddev = cp.sqrt(normalized.var(axis=0))
    normalized /= stddev
    del stddev
    
    return normalized.clip(a_max=max_value)

    
    
    
def _regress_out_chunk(X, y):
    """
    Performs a data_cunk.shape[1] number of local linear regressions,
    replacing the data in the original chunk w/ the regressed result.

    Parameters
    ----------

    X : cupy.ndarray of shape (n_cells, 3)
        Matrix of regressors

    y : cupy.sparse.spmatrix of shape (n_cells,)
        Sparse matrix containing a single column of the cellxgene matrix

    Returns
    -------

    dense_mat : cupy.ndarray of shape (n_cells,)
        Adjusted column
    """
    y_d = y.todense()
    
    lr = LinearRegression(fit_intercept=False, output_type="cupy")
    lr.fit(X, y_d, convert_dtype=True)
    return y_d.reshape(y_d.shape[0],) - lr.predict(X).reshape(y_d.shape[0])
    



def regress_out(adata, keys, verbose=False):

    """
    Use linear regression to adjust for the effects of unwanted noise
    and variation. 
    Parameters
    ----------

    adata
        The annotated data matrix.
    keys
        Keys for numerical observation annotation on which to regress on.

    verbose : bool
        Print debugging information

    Returns
    -------
    outputs : cupy.ndarray
        Adjusted dense matrix
    
    """
    
    normalized = cp.sparse.csc_matrix(adata.X)
    
    dim_regressor= 2
    if type(keys)is list:
        dim_regressor = len(keys)+1
    
    regressors = cp.ones((adata.n_obs*dim_regressor)).reshape((adata.n_obs, dim_regressor), order="F")
    if dim_regressor==2:
        regressors[:, 1] = cp.array(adata.obs[keys]).ravel()
    else:
        for i in range(dim_regressor-1):
            regressors[:, i+1] = cp.array(adata.obs[keys[i]]).ravel()
    
    outputs = cp.empty(normalized.shape, dtype=normalized.dtype, order="F")
    
    for i in range(normalized.shape[1]):
        if verbose and i % 500 == 0:
            print("Regressed %s out of %s" %(i, normalized.shape[1]))
        X = regressors
        y = normalized[:,i]
        outputs[:, i] = _regress_out_chunk(X, y)
    return outputs
    





def select_groups(labels, groups_order_subset='all'):
    adata_obs_key = labels
    groups_order = labels.cat.categories
    groups_masks = cp.zeros(
        (len(labels.cat.categories), len(labels.cat.codes)), dtype=bool
    )
    for iname, name in enumerate(labels.cat.categories.to_pandas()):
        # if the name is not found, fallback to index retrieval
        if labels.cat.categories[iname] in labels.cat.codes:
            mask = labels.cat.categories[iname] == labels.cat.codes
        else:
            mask = iname == labels.cat.codes
        groups_masks[iname] = mask.values
    groups_ids = list(range(len(groups_order)))
    if groups_order_subset != 'all':
        groups_ids = []
        for name in groups_order_subset:
            groups_ids.append(
                cp.where(cp.array(labels.cat.categories.to_array().astype("int32")) == int(name))[0][0]
            )
        if len(groups_ids) == 0:
            # fallback to index retrieval
            groups_ids = cp.where(
                cp.in1d(
                    cp.arange(len(labels.cat.categories)).astype(str),
                    cp.array(groups_order_subset),
                )
            )[0]
            
        groups_ids = [groups_id.item() for groups_id in groups_ids]
        groups_masks = groups_masks[groups_ids]
        groups_order_subset = labels.cat.categories[groups_ids].to_array().astype(int)
    else:
        groups_order_subset = groups_order.to_array()
    return groups_order_subset, groups_masks


def rank_genes_groups(
    X,
    labels,  # louvain results
    var_names,
    groups=None,
    reference='rest',
    n_genes=100,
    **kwds,
):

    """
    Rank genes for characterizing groups.

    Parameters
    ----------

    X : cupy.ndarray of shape (n_cells, n_genes)
        The cellxgene matrix to rank genes

    labels : cudf.Series of size (n_cells,)
        Observations groupings to consider

    var_names : cudf.Series of size (n_genes,)
        Names of genes in X

    groups : Iterable[str] (default: 'all')
        Subset of groups, e.g. ['g1', 'g2', 'g3'], to which comparison
        shall be restricted, or 'all' (default), for all groups.

    reference : str (default: 'rest')
        If 'rest', compare each group to the union of the rest of the group.
        If a group identifier, compare with respect to this group.

    n_genes : int (default: 100)
        The number of genes that appear in the returned tables.
    """

    #### Wherever we see "adata.obs[groupby], we should just replace w/ the groups"

    import time
    
    start = time.time()
    
    # for clarity, rename variable
    if groups == 'all':
        groups_order = 'all'
    elif isinstance(groups, (str, int)):
        raise ValueError('Specify a sequence of groups')
    else:
        groups_order = list(groups)
        if isinstance(groups_order[0], int):
            groups_order = [str(n) for n in groups_order]
        if reference != 'rest' and reference not in set(groups_order):
            groups_order += [reference]
    if (
        reference != 'rest'
        and reference not in set(labels.cat.categories)
    ):
        cats = labels.cat.categories.tolist()
        raise ValueError(
            f'reference = {reference} needs to be one of groupby = {cats}.'
        )

    groups_order, groups_masks = select_groups(labels, groups_order)
    
    original_reference = reference
    
    n_vars = len(var_names)

    # for clarity, rename variable
    n_genes_user = n_genes
    # make sure indices are not OoB in case there are less genes than n_genes
    if n_genes_user > X.shape[1]:
        n_genes_user = X.shape[1]
    # in the following, n_genes is simply another name for the total number of genes
    n_genes = X.shape[1]

    n_groups = groups_masks.shape[0]
    ns = cp.zeros(n_groups, dtype=int)
    for imask, mask in enumerate(groups_masks):
        ns[imask] = cp.where(mask)[0].size
    if reference != 'rest':
        ireference = cp.where(groups_order == reference)[0][0]
    reference_indices = cp.arange(n_vars, dtype=int)

    rankings_gene_scores = []
    rankings_gene_names = []

    # Perform LogReg
        
    # if reference is not set, then the groups listed will be compared to the rest
    # if reference is set, then the groups listed will be compared only to the other groups listed
    from cuml.linear_model import LogisticRegression
    reference = groups_order[0]
    if len(groups) == 1:
        raise Exception('Cannot perform logistic regression on a single cluster.')
    grouping_mask = labels.astype('int').isin(cudf.Series(groups_order))
    grouping = labels.loc[grouping_mask]
    
    X = X[grouping_mask.values, :]  # Indexing with a series causes issues, possibly segfault
    y = labels.loc[grouping]
    
    clf = LogisticRegression(**kwds)
    clf.fit(X.get(), grouping.to_array().astype('float32'))
    scores_all = cp.array(clf.coef_).T
    
    for igroup, group in enumerate(groups_order):
        if len(groups_order) <= 2:  # binary logistic regression
            scores = scores_all[0]
        else:
            scores = scores_all[igroup]

        partition = cp.argpartition(scores, -n_genes_user)[-n_genes_user:]
        partial_indices = cp.argsort(scores[partition])[::-1]
        global_indices = reference_indices[partition][partial_indices]
        rankings_gene_scores.append(scores[global_indices].get())  ## Shouldn't need to take this off device
        rankings_gene_names.append(var_names[global_indices].to_pandas())
        if len(groups_order) <= 2:
            break

    groups_order_save = [str(g) for g in groups_order]
    if (len(groups) == 2):
        groups_order_save = [g for g in groups_order if g != reference]
    
    print("Ranking took (GPU): " + str(time.time() - start))
    
    start = time.time()
    
    scores = np.rec.fromarrays(
        [n for n in rankings_gene_scores],
        dtype=[(rn, 'float32') for rn in groups_order_save],
    )
    
    names = np.rec.fromarrays(
        [n for n in rankings_gene_names],
        dtype=[(rn, 'U50') for rn in groups_order_save],
    )
    
    print("Preparing output np.rec.fromarrays took (CPU): " + str(time.time() - start))
    print("Note: This operation will be accelerated in a future version")
    
    return scores, names, original_reference


def leiden(adata, resolution=1.0):
    """
    Performs Leiden Clustering using cuGraph
    Parameters
    ----------
    adata : annData object with 'neighbors' field.
       
    resolution : float, optional (default: 1)
        A parameter value controlling the coarseness of the clustering.
        Higher values lead to more clusters.
    """
    # Adjacency graph
    adjacency = adata.uns['neighbors']['connectivities']
    offsets = cudf.Series(adjacency.indptr)
    indices = cudf.Series(adjacency.indices)
    g = cugraph.Graph()
    if hasattr(g, 'add_adj_list'):
        g.add_adj_list(offsets, indices, None)
    else:
        g.from_cudf_adjlist(offsets, indices, None)
    
    # Cluster
    leiden_parts, _ = cugraph.leiden(g,resolution = resolution)
    
    # Format output
    clusters = leiden_parts.to_pandas().sort_values('vertex')[['partition']].to_numpy().ravel()
    clusters = pd.Categorical(clusters)
    
    adata.obs['leiden'] = clusters
    
def kmeans(adata, n_clusters =8, random_state= 42):
    """
    KMeans is a basic but powerful clustering method which is optimized via
Expectation Maximization. 

    Parameters
    ----------
    adata: adata object with `.obsm['X_pca']`
    
    n_clusters: int (default:8)
        Number of clusters to compute
        
    random_state: float (default: 42)
        if you want results to be the same when you restart Python, select a
    state.
    
    """

    
    kmeans_out = KMeans(n_clusters=n_clusters, random_state=random_state).fit(adata.obsm['X_pca'])
    adata.obs['kmeans'] = kmeans_out.labels_.astype(str)

def pca(adata, n_comps = 50):
    """
    Performs PCA using the cuML decomposition function
    
    Parameters
    ----------
    adata : annData object
    
    n_comps: int (default: 50)
        Number of principal components to compute. Defaults to 50
    
    Returns
    
    else adds fields to `adata`:

    `.obsm['X_pca']`
         PCA representation of data.  
    `.uns['pca']['variance_ratio']`
         Ratio of explained variance.
    `.uns['pca']['variance']`
         Explained variance, equivalent to the eigenvalues of the
         covariance matrix.
    """
    pca_func = PCA(n_components=n_comps, output_type="numpy")
    adata.obsm["X_pca"] = pca_func.fit_transform(adata.X)
    adata.uns['pca'] ={'variance':pca_func.explained_variance_, 'variance_ratio':pca_func.explained_variance_ratio_}
    
    
def tsne(adata, n_pcs,perplexity = 30, early_exaggeration = 12,learning_rate =1000):
    """
    Performs t-distributed stochastic neighborhood embedding (tSNE) using cuML libraray. Variable description adapted from scanpy and default are the same
    
    Parameters
    ---------
    adata: adata object with `.obsm['X_pca']`
    
    n_pcs: int
        use this many PCs
    
    perplexity: float (default: 30)
        The perplexity is related to the number of nearest neighbors that is used in other manifold learning algorithms. Larger datasets usually require a larger perplexity. Consider selecting a value between 5 and 50. The choice is not extremely critical since t-SNE is quite insensitive to this parameter.
    
    early_exaggeration : float (default:12)
        Controls how tight natural clusters in the original space are in the embedded space and how much space will be between them. For larger values, the space between natural clusters will be larger in the embedded space. Again, the choice of this parameter is not very critical. If the cost function increases during initial optimization, the early exaggeration factor or the learning rate might be too high.
    
    learning_rate : float (default:1000)
        Note that the R-package “Rtsne” and cuML uses a default of 200. The learning rate can be a critical parameter. It should be between 100 and 1000. If the cost function increases during initial optimization, the early exaggeration factor or the learning rate might be too high. If the cost function gets stuck in a bad local minimum increasing the learning rate helps sometimes.


    """
    
    adata.obsm['X_tsne'] = TSNE(perplexity=perplexity, early_exaggeration=early_exaggeration,learning_rate=learning_rate).fit_transform(adata.obsm["X_pca"][:,:n_pcs])

    
def plt_scatter(cudata, x, y, save = None, show =True, dpi =300):
    """
    Violin plot.
    Wraps :func:`seaborn.scaterplot` for :class:`~cunnData.cunnData`. This plotting function so far is really basic and doesnt include all the features form sc.pl.scatter.
    
    Parameters
    ---------
    cudata:
        cunnData object
    
    x:
        Keys for accessing variables of fields of `.obs`.
    
    y:
        Keys for accessing variables of fields of `.obs`.

    
    save: str default(None (no plot will be saved))
        file name to save plot as in ./figures
        
    show: boolean (default: True)
        if you want to display the plot
    
    dpi: int (default: 300)
        The resolution in dots per inch for save
    
    Returns
    ------
    nothing
    
    """
    fig,ax = plt.subplots()
    sns.scatterplot(data=cudata.obs, x=x, y=y, color='k')
    if save:
        os.makedirs("./figures/",exist_ok=True)
        fig_path = "./figures/"+save
        plt.savefig(fig_path, dpi=dpi ,bbox_inches = 'tight')
    if show is False:
        plt.close()

        
def plt_violin(cudata, key, group_by=None, size =1, save = None, show =True, dpi =300):
    """
    Violin plot.
    Wraps :func:`seaborn.violinplot` for :class:`~cunnData.cunnData`. This plotting function so far is really basic and doesnt include all the features form sc.pl.violin.
    
    Parameters
    ---------
    cudata:
        cunnData object
    
    key:
        Keys for accessing variables of fields of `.obs`.
    
    group_by:
        The key of the observation grouping to consider.(e.g batches)
    
    size:
        pt_size for stripplot if 0 no strip plot will be shown.
    
    save: str default(None (no plot will be saved))
        file name to save plot as in ./figures
        
    show: boolean (default: True)
        if you want to display the plot
    
    dpi: int (default: 300)
        The resolution in dots per inch for save
    
    Returns
    ------
    nothing
    
    """
    fig,ax = plt.subplots()
    ax = sns.violinplot(data=cudata.obs, y=key,scale='width',x= group_by, inner = None)
    if size:
        ax = sns.stripplot(data=cudata.obs, y=key,x= group_by, color='k', size= 1, dodge = True, jitter = True)
    if save:
        os.makedirs("./figures/",exist_ok=True)
        fig_path = "./figures/"+save
        plt.savefig(fig_path, dpi=dpi ,bbox_inches = 'tight')
    if show is False:
        plt.close()