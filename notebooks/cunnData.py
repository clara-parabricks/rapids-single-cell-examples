import cupy as cp
import cudf
import cugraph
import anndata

import numpy as np
import pandas as pd
import scipy
import math
from scipy import sparse
from typing import Any, Union, Optional

from cuml.linear_model import LinearRegression


class cunnData:
    """
    The cunnData objects can be used as an AnnData replacement for the inital preprocessing of single cell Datasets. It replaces some of the most common preprocessing steps within scanpy for annData objects.
    It can be initalized with a preexisting annData object or with a countmatrix and seperate Dataframes for var and obs. Index of var will be used as gene_names. Initalization with an AnnData object is advised.
    """
    shape = tuple
    nnz = int
    genes = cudf.Series
    def __init__(
        self,
        X: Optional[Union[np.ndarray,sparse.spmatrix, cp.array, cp.sparse.csr_matrix]] = None,
        obs: Optional[pd.DataFrame] = None,
        var: Optional[pd.DataFrame] = None,
        adata: Optional[anndata.AnnData] = None):
            if adata:
                self.X = cp.sparse.csr_matrix(adata.X, dtype=cp.float32)
                self.shape = self.X.shape
                self.nnz = self.X.nnz
                self.genes = cudf.Series(adata.var_names)
                self.obs = adata.obs.copy()
                self.var = adata.var.copy()  
            else:
                self.X = cp.sparse.csr_matrix(X)
                self.shape = self.X.shape
                self.nnz = self.X.nnz
                genes = cudf.Series(var.index)
                self.obs = obs
                self.var = var
                
  
    def to_AnnData(self):
        """
        Takes the cunnData object and creates an AnnData object
        
        Returns
        -------
            annData object
        
        """
        adata = anndata.AnnData(self.X.get())
        adata.obs = self.obs.copy()
        adata.var = self.var.copy()
        return adata

    def filter_genes(self, min_cells = 3, batchsize = None, verbose =True):
        """
        Filters out genes that expressed in less than a specified number of cells

        Parameters
        ----------
        
            min_cells: int (default = 3)
                Genes containing a number of cells below this value will be filtered
        
            batchsize: int (default: None)
                Number of rows to be processed together This can be adjusted for performance to trade-off memory use.
        
            verbose: bool
                Print number of discarded genes
            
        Returns
        -------
            filtered cunndata object inplace for genes less than the threshhold
        
        """
        if batchsize:
            n_batches = math.ceil(self.X.shape[0] / batchsize)
            filter_matrix = cp.zeros(shape=(n_batches,self.X.shape[1]))
            for batch in range(n_batches):
                batch_size = batchsize
                start_idx = batch * batch_size
                stop_idx = min(batch * batch_size + batch_size, self.X.shape[0])
                arr_batch = self.X[start_idx:stop_idx]
                thr = cp.diff(arr_batch.tocsc().indptr)
                thr = thr.ravel()
                filter_matrix[batch,:]=thr
            thr = cp.asarray(filter_matrix.sum(axis=0) >= min_cells).ravel()
            thr = cp.where(thr)[0]
            if verbose:
                print(f"filtered out {self.X.shape[1]-len(thr)} genes that are detected in less than {min_cells} cells")
            self.X =self.X.tocsc()
            self.X = self.X[:, thr]
            self.shape = self.X.shape
            self.nnz = self.X.nnz
            self.X = self.X.tocsr()
            self.genes = self.genes[thr]
            self.genes = self.genes.reset_index(drop=True)
            self.var = self.var.iloc[cp.asnumpy(thr)]
                
                
        else:
            self.X =self.X.tocsc()
            thr = cp.diff(self.X.indptr).ravel()
            thr = cp.where(thr >= min_cells)[0]
            if verbose:
                print(f"filtered out {self.X.shape[1]-len(thr)} genes that are detected in less than {min_cells} cells")
            self.X = self.X[:, thr]
            self.shape = self.X.shape
            self.nnz = self.X.nnz
            self.X = self.X.tocsr()
            self.genes = self.genes[thr]
            self.genes = self.genes.reset_index(drop=True)
            self.var = self.var.iloc[cp.asnumpy(thr)]

        
    def caluclate_qc(self, qc_vars = None, batchsize = None):
        """
        Calculates basic qc Parameters. Calculates number of genes per cell (n_genes) and number of counts per cell (n_counts).
        Loosly based on calculate_qc_metrics from scanpy [Wolf et al. 2018]. Updates .obs with columns with qc data.
        
        Parameters
        ----------
        qc_vars: str, list (default: None)
            Keys for boolean columns of .var which identify variables you could want to control for (e.g. Mito). Run flag_gene_family first
            
        batchsize: int (default: None)
            Number of rows to be processed together. This can be adjusted for performance to trade-off memory use.
            
        Returns
        -------
        adds the following columns in .obs
        n_counts
            number of counts per cell
        n_genes
            number of genes per cell
        for qc_var in qc_vars
            total_qc_var
                number of counts per qc_var (e.g total counts mitochondrial genes)
            percent_qc_vars
                
                Proportion of counts of qc_var (percent of counts mitochondrial genes)
        
        """      
        if batchsize:
            n_batches = math.ceil(self.X.shape[0] / batchsize)
            n_genes = []
            n_counts = []
            if qc_vars:
                if type(qc_vars) is str:
                    qc_var_total = []
                    
                elif type(qc_vars) is list:
                    qc_var_total = []
                    for i in range(len(qc_vars)):
                        my_list = []
                        qc_var_total.append(my_list)
                        
            for batch in range(n_batches):
                batch_size = batchsize
                start_idx = batch * batch_size
                stop_idx = min(batch * batch_size + batch_size, self.X.shape[0])
                arr_batch = self.X[start_idx:stop_idx]
                n_genes.append(cp.diff(arr_batch.indptr).ravel().get())
                n_counts.append(arr_batch.sum(axis=1).ravel().get())
                if qc_vars:
                    if type(qc_vars) is str:
                        qc_var_total.append(arr_batch[:,self.var[qc_vars]].sum(axis=1).ravel().get())

                    elif type(qc_vars) is list:
                        for i in range(len(qc_vars)):
                             qc_var_total[i].append(arr_batch[:,self.var[qc_vars[i]]].sum(axis=1).ravel().get())
                        
                
            self.obs["n_genes"] = np.concatenate(n_genes)
            self.obs["n_counts"] = np.concatenate(n_counts)
            if qc_vars:
                if type(qc_vars) is str:
                    self.obs["total_"+qc_vars] = np.concatenate(qc_var_total)
                    self.obs["percent_"+qc_vars] =self.obs["total_"+qc_vars]/self.obs["n_counts"]*100
                elif type(qc_vars) is list:
                    for i in range(len(qc_vars)):
                        self.obs["total_"+qc_vars[i]] = np.concatenate(qc_var_total[i])
                        self.obs["percent_"+qc_vars[i]] =self.obs["total_"+qc_vars[i]]/self.obs["n_counts"]*100
        else:
            self.obs["n_genes"] = cp.asnumpy(cp.diff(self.X.indptr)).ravel()
            self.obs["n_counts"] = cp.asnumpy(self.X.sum(axis=1)).ravel()
            if qc_vars:
                if type(qc_vars) is str:
                    self.obs["total_"+qc_vars]=cp.asnumpy(self.X[:,self.var[qc_vars]].sum(axis=1))
                    self.obs["percent_"+qc_vars]=self.obs["total_"+qc_vars]/self.obs["n_counts"]*100

                elif type(qc_vars) is list:
                    for qc_var in qc_vars:
                        self.obs["total_"+qc_var]=cp.asnumpy(self.X[:,self.var[qc_var]].sum(axis=1))
                        self.obs["percent_"+qc_var]=self.obs["total_"+qc_var]/self.obs["n_counts"]*100
    
    def flag_gene_family(self, gene_family_name = str, gene_family_prefix = None, gene_list= None):
        """
        Flags a gene or gene_familiy in .var with boolean. (e.g all mitochondrial genes).
        Please only choose gene_family prefix or gene_list
        
        Parameters
        ----------
        gene_family_name: str
            name of colums in .var where you want to store informationa as a boolean
            
        gene_family_prefix: str
            prefix of the gene familiy (eg. mt- for all mitochondrial genes in mice)
            
        gene_list: list
            list of genes to flag in .var
        
        Returns
        -------
        adds the boolean column in .var 
        
        """
        if gene_family_prefix:
            self.var[gene_family_name] = cp.asnumpy(self.genes.str.startswith(gene_family_prefix)).ravel()
        if gene_list:
            self.var[gene_family_name] = cp.asnumpy(self.genes.isin(gene_list)).ravel()
    
    def filter_cells(self, qc_var, min_count=None, max_count=None, batchsize = None,verbose=True):
        """
        Filter cells that have greater than a max number of genes or less than
        a minimum number of a feature in a given .obs columns. Can so far only be used for numerical columns.
        It is recommended to run `calculated_qc` before using this function. You can run this function on n_genes or n_counts before running `calculated_qc`.
        
        Parameters
        ----------
        qc_var: str
            column in .obs with numerical entries to filter against
            
        min_count : float
            Lower bound on number of a given feature to keep cell

        max_count : float
            Upper bound on number of a given feature to keep cell
        
        batchsize: int (default: None)
            only needed if you run `filter_cells` before `calculate_qc` on 'n_genes' or 'n_counts'. Number of rows to be processed together. This can be adjusted for performance to trade-off memory use.
            
        verbose: bool (default: True)
            Print number of discarded cells
        
        Returns
        -------
        a filtered cunnData object inplace
        
        """
        if qc_var in self.obs.keys(): 
            inter = np.array
            if min_count is not None and max_count is not None:
                inter=np.where((self.obs[qc_var] < max_count) &  (min_count< self.obs[qc_var]))[0]
            elif min_count is not None:
                inter=np.where(self.obs[qc_var] > min_count)[0]
            elif max_count is not None:
                inter=np.where(self.obs[qc_var] < max_count)[0]
            else:
                print(f"Please specify a cutoff to filter against")
            if verbose:
                print(f"filtered out {self.obs.shape[0]-inter.shape[0]} cells")
            self.X = self.X[inter,:]
            self.shape = self.X.shape
            self.nnz = self.X.nnz
            self.obs = self.obs.iloc[inter]
        elif qc_var in ['n_genes','n_counts']:
            print(f"Running calculate_qc for 'n_genes' or 'n_counts'")
            self.caluclate_qc(batchsize=batchsize)
            inter = np.array
            if min_count is not None and max_count is not None:
                inter=np.where((self.obs[qc_var] < max_count) &  (min_count< self.obs[qc_var]))[0]
            elif min_count is not None:
                inter=np.where(self.obs[qc_var] > min_count)[0]
            elif max_count is not None:
                inter=np.where(self.obs[qc_var] < max_count)[0]
            else:
                print(f"Please specify a cutoff to filter against")
            if verbose:
                print(f"filtered out {self.obs.shape[0]-inter.shape[0]} cells")
            self.X = self.X[inter,:]
            self.shape = self.X.shape
            self.nnz = self.X.nnz
            self.obs = self.obs.iloc[inter]
        else:
            print(f"Please check qc_var.")
            

        
    def normalize_total(self, target_sum):
        """
        Normalizes rows in matrix so they sum to `target_sum`

        Parameters
        ----------

        target_sum : int
            Each row will be normalized to sum to this value
        
        
        Returns
        -------
        
        a normalized sparse Matrix to a specified target sum
        
        """
        csr_arr = self.X
        mul_kernel = cp.RawKernel(r'''
            extern "C" __global__
            void mul_kernel(const int *indptr, float *data, 
                            int nrows, int tsum) {
                int row = blockDim.x * blockIdx.x + threadIdx.x;

                if(row >= nrows)
                    return;

                float scale = 0.0;
                int start_idx = indptr[row];
                int stop_idx = indptr[row+1];

                for(int i = start_idx; i < stop_idx; i++)
                    scale += data[i];

                if(scale > 0.0) {
                    scale = tsum / scale;
                    for(int i = start_idx; i < stop_idx; i++)
                        data[i] *= scale;
                }
            }
            ''', 'mul_kernel')

        mul_kernel((math.ceil(csr_arr.shape[0] / 32.0),), (32,),
                       (csr_arr.indptr,
                        csr_arr.data,
                        csr_arr.shape[0],
                       int(target_sum)))

        self.X = csr_arr
    
    def log1p(self):
        """
        Calculated the natural logarithm of one plus the sparse marttix, element-wise inlpace in cunnData object.
        """
        self.X = self.X.log1p()