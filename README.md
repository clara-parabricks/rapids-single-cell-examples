# GPU-Accelerated Single-Cell Genomics Analysis with RAPIDS

This repository contains example notebooks demonstrating how to use [RAPIDS](https://rapids.ai) for GPU-accelerated analysis of single-cell sequencing data.

RAPIDS is a suite of open-source Python libraries that can speed up data science workflows using GPU acceleration.Starting from a single-cell count matrix, RAPIDS libraries can be used to perform data processing, dimensionality reduction, clustering, visualization, and comparison of cell clusters.

Several of our examples are inspired by the [Scanpy tutorials](https://scanpy.readthedocs.io/en/stable/tutorials.html) and based upon the [AnnData](https://anndata.readthedocs.io/en/latest/index.html) format. Currently, we provide examples for scRNA-seq and scATAC-seq, and we have scaled up to 1 million cells. We also show how to create GPU-powered interactive, in-browser visualizations to explore single-cell datasets.

Dataset sizes for single-cell genomics studies are increasing, presently reaching millions of cells. With RAPIDS, it becomes easy to analyze large datasets interactively and in real time, enabling faster scientific discoveries.

## Installation

### Docker container
A container with all dependencies, notebooks and source code is available at https://hub.docker.com/r/claraparabricks/single-cell-examples_rapids_cuda11.0.

Please execute the following commands to start the notebook and follow the URL in the log to open Jupyter web application.

```bash
docker pull claraparabricks/single-cell-examples_rapids_cuda11.0

docker run --gpus all --rm -v /mnt/data:/data claraparabricks/single-cell-examples_rapids_cuda11.0
```

### conda
All dependencies for these examples can be installed with conda. CUDA versions 10.1 and higher are supported currently.

```bash
conda env create --name rapidgenomics -f conda/rapidgenomics_cuda10.2.yml
conda activate rapidgenomics
python -m ipykernel install --user --display-name "Python (rapidgenomics)"
```
If installing for a system running a CUDA 10.1 driver, use `conda/rapidgenomics_cuda10.1.yml`. For CUDA 11.0, use `conda/rapidgenomics_cuda11.0.yml`

After installing the necessary dependencies, you can just run `jupyter lab`.

### Launch Script
Lanuch script (./launch) can be used to start example notebooks either on a host or in a docker container. This script prepares the environment and acquires the dataset for the examples.

```bash
# rapids-single-cell-examples$ ./launch -h
usage: launch <command> [<args>]

Following commands are wrapped by this tool:
   container  : Start Jupyter notebook in a container
   host       : Start Jupyter notebook on the host
   dataset    : Download dataset
   execute    : Execute an example
   create_env : Create conda environment for an example

To execute 'hlca_lung' example in container, please execute the following command:
./launch container -d /path/to/store/dataset -e hlca_lung

Example launcher

positional arguments:
  command     Subcommand to run

optional arguments:
  -h, --help  show this help message and exit
```

```./launch host``` can be used to create a conda environment for executing any of the examples. To list all supported examples, please execute ```./launch host -h```.

```./launch container``` can be used to setup a container for the example.

```./launch execute```, can be used to run an example in the background. Results are saved inplace.


## Configuration


Unified Virtual Memory (UVM) can be used to [oversubscribe](https://developer.nvidia.com/blog/beyond-gpu-memory-limits-unified-memory-pascal/) your GPU memory so that chunks of data will be automatically offloaded to main memory when necessary. This is a great way to explore data without having to worry about out of memory errors, but it does degrade performance in proportion to the amount of oversubscription. UVM is enabled by default in these examples and can be enabled/disabled in any RAPIDS workflow with the following:
```python
import cupy as cp
import rmm
rmm.reinitialize(managed_memory=True)
cp.cuda.set_allocator(rmm.rmm_cupy_allocator)
```

RAPIDS provides a [GPU Dashboard](https://medium.com/rapids-ai/gpu-dashboards-in-jupyter-lab-757b17aae1d5), which contains useful tools to monitor GPU hardware right in Jupyter.


## Citation

If you use this code, please cite: <a href="https://zenodo.org/badge/latestdoi/265649968"><img src="https://zenodo.org/badge/265649968.svg" alt="DOI"></a>



## Example 1: Single-cell RNA-seq of 70,000 Human Lung Cells

<img align="left" width="240" height="200" src="https://github.com/clara-parabricks/rapids-single-cell-examples/blob/master/images/70k_lung.png?raw=true">

We use RAPIDS to accelerate the analysis of a ~70,000-cell single-cell RNA sequencing dataset from human lung cells. This example includes preprocessing, dimension reduction, clustering, visualization and gene ranking.

### Example Dataset

The dataset is from [Travaglini et al. 2020](https://www.biorxiv.org/content/10.1101/742320v2). If you wish to run the example notebook using the same data, use the following command to download the count matrix for this dataset and store it in the `data` folder:

```bash
wget -P <path to this repository>/data https://rapids-single-cell-examples.s3.us-east-2.amazonaws.com/krasnow_hlca_10x.sparse.h5ad
```

### Example Code

Follow this [Jupyter notebook](notebooks/hlca_lung_gpu_analysis.ipynb) for RAPIDS analysis of this dataset. In order for the notebook to run, the file [rapids_scanpy_funcs.py](notebooks/rapids_scanpy_funcs.py) needs to be in the same folder as the notebook.

We provide a second notebook with the CPU version of this analysis [here](notebooks/hlca_lung_cpu_analysis.ipynb).

### Acceleration

We report the runtime of these notebooks on various GCP instances below. All runtimes are given in seconds. Acceleration is given in parentheses. Benchmarking was performed on Dec 16, 2020.

| Step                         | CPU <br> e2-standard-32 <br> 32 vCPUs | GPU <br> n1-standard-16 <br> T4 16 GB GPU <br> (Acceleration)  | GPU <br> n1-highmem-8 <br> Tesla V100 16 GB GPU <br> (Acceleration) | GPU <br> a2-highgpu-1g <br> Tesla A100 40GB GPU <br> (Acceleration) |
|------------------------------|---------------------------|---------------------------|---------------|--------------|
| Preprocessing                | 351                       | 87       (4x)             | 78   (5x)     | 91   (4x)    |
| PCA                          | 5.7                       | 4.6      (1.2x)           | 3.2  (2x)     | 2.7  (2x)    |
| t-SNE                        | 235                       | 3.8      (62x)            | 1.9  (124x)   | 2.2  (107x)  |
| k-means (single iteration)   | 17.6                      | 0.55     (32x)            | 0.14 (126x)   | 0.09 (196x)  |
| KNN                          | 39                        | 20.6     (2x)             | 20.9 (2x)     | 5.3  (7x)    |
| UMAP                         | 46                        | 0.97     (47x)            | 0.52 (88x)    | 0.63 (73x)   |
| Louvain clustering           | 16.9                      | 0.22     (77x)            | 0.19 (89x)    | 0.14 (121x)  |
| Leiden clustering            | 16.5                      | 0.15     (110x)           | 0.12 (138x)   | 0.12 (138x)  |
| Differential Gene Expression | 108                       | 6.9      (16x)            | 2.5  (43x)    | 2.0  (54x)   |
| Re-analysis of subgroup      | 28                        | 5.1      (5x)             | 4.3  (7x)     | 4.1  (7x)    |
| End-to-end notebook run      | 883                       | 154                       | 142           | 125          |
| Price ($/hr)                 | 1.073                     | 1.110                     | 2.953         | 4.00         |
| Total cost ($)               | 0.263                     | 0.047                     | 0.116         | 0.139        |



## Example 2: Single-cell RNA-seq of 1.3 Million Mouse Brain Cells

<img align="left" width="240" height="200" src="https://github.com/clara-parabricks/rapids-single-cell-examples/blob/master/images/1M_brain.png?raw=true">

We demonstrate the use of RAPIDS along with Dask to accelerate the analysis of single-cell RNA-seq data from 1.3 million cells. This example includes preprocessing, dimension reduction, clustering and visualization. Dask is used to load and preprocess the dataset in batches.

This example relies heavily on UVM and a few of the operations oversubscribed a 32GB V100 GPU on a DGX1. While this example should work on any GPU built on the Pascal architecture or newer, you will want to make sure there is enough main memory available. Oversubscribing a GPU by more than a factor of 2x can cause thrashing in UVM, which can ultimately lead to the notebook freezing.

### Example Dataset

The dataset was made publicly available by 10X Genomics. Use the following command to download the count matrix for this dataset and store it in the `data` folder:

```bash
wget -P <path to this repository>/data https://rapids-single-cell-examples.s3.us-east-2.amazonaws.com/1M_brain_cells_10X.sparse.h5ad
```

### Example Code

Follow this [Jupyter notebook](notebooks/1M_brain_gpu_analysis_uvm.ipynb) for RAPIDS analysis of this dataset. In order for the notebook to run, the files [rapids_scanpy_funcs.py](notebooks/rapids_scanpy_funcs.py) and [utils.py](notebooks/utils.py) need to be in the same folder as the notebook.

We provide a second notebook with the CPU version of this analysis [here](notebooks/1M_brain_cpu_analysis.ipynb).

### Acceleration

We report the runtime of these notebooks on various GCP instances below. All runtimes are given in seconds. Acceleration is given in parentheses. Benchmarking was performed on Dec 16, 2020. Note: this section is out of date and will be revised shortly.

| Step                         | CPU <br> n1-highmem-32 <br> 32 vCPUs | GPU <br> n1-highmem-16 <br> T4 16 GB GPU <br> (Acceleration)  | GPU <br> n1-highmem-16 <br> Tesla V100 16 GB GPU <br> (Acceleration) | GPU <br> a2-highgpu-1g <br> Tesla A100 40GB GPU <br> (Acceleration) |
|------------------------------|---------------------------|----------------------------|-------------------|--------------|
| Preprocessing                | 1715                      | 756  (2.3x)                | 308  (6x)         | 201  (9x)    |
| PCA                          | 29.2                      | 28   (1.04x)               | 24   (1.2x)       | 11.4 (2.6x)  |
| t-SNE                        | 4990                      | 128  (39x)                 | 39   (128x)       | 28   (178x)  |
| k-means (single iteration)   | 126                       | 13.7 (9x)                  | 2.7  (47x)        | 1.9  (66x)   |
| KNN                          | 185                       | 150  (1.2x)                | 89   (2.1x)       | 46   (4x)    |
| UMAP                         | 1307                      | 79   (17x)                 | 18.8 (70x)        | 13.4 (98x)   |
| Louvain clustering           | 905                       | 5.0  (181x)                | 2.8  (323x)       | 1.9  (476x)  |
| Leiden clustering            | 3061                      | 4.4  (696x)                | 2.0  (1531x)      | 1.4  (2186x) |
| Re-analysis of subgroup      | 159                       | 15.8 (10x)                 | 13   (12x)        | 9.3  (17x)   |
| End-to-end notebook run      | 12708                     | 1399                       | 702               | 502          |
| Price ($/hr)                 | 1.893                     | 1.296                      | 5.906             | 4.00         |
| Total cost ($)               | 6.682                     | 0.504                      | 1.164             | 0.553        |


## Example 3: GPU-based Interactive Visualization of 70,000 Human Lung Cells (beta version)

![Interactive browser Demo](images/viz3-2.gif)

We demonstrate how to use RAPIDS, Scanpy and Plotly Dash to create an interactive dashboard where we visualize a single-cell RNA-sequencing dataset. Within the interactive dashboard, we can cluster, visualize, and compare any selected groups of cells.

### Installation

Additional dependencies are needed for this example. Follow these instructions for conda installation:

```bash
conda env create --name rapidgenomics-viz -f conda/rapidgenomics_cuda10.2.viz.yml
conda activate rapidgenomics-viz
python -m ipykernel install --user --display-name "Python (rapidgenomics-viz)"
```

After installing the necessary dependencies, you can just run `jupyter lab`.

### Example Dataset

The dataset used here is the same as in example 1.

### Example Code

Follow this [Jupyter notebook](notebooks/hlca_lung_gpu_analysis-visualization.ipynb) to create the interactive visualization. In order for the notebook to run, the files [rapids_scanpy_funcs.py](notebooks/rapids_scanpy_funcs.py) and [visualize.py](notebooks/visualize.py) need to be in the same folder as the notebook.

## Example 4: Droplet Single-cell ATAC-seq of 60K Bone Marrow Cells

<img align="left" width="240" height="200" src="https://github.com/clara-parabricks/rapids-single-cell-examples/blob/master/images/60k_bmmc_dsciATAC.png?raw=true">

We demonstrate the use of RAPIDS to accelerate the analysis of single-cell ATAC-seq data from 60,495 cells. We start with the peak-cell matrix, then perform peak selection, normalization, dimensionality reduction, clustering, and visualization. We also visualize regulatory activity at marker genes and compute differential peaks.

### Example Dataset

The dataset is taken from [Lareau et al., Nat Biotech 2019](https://www.nature.com/articles/s41587-019-0147-6). We processed the dataset to include only cells in the 'Resting' condition and peaks with nonzero coverage. Use the following command to download (1) the processed peak-cell count matrix for this dataset (.h5ad), (2) the set of nonzero peak names (.npy), and (3) the cell metadata (.csv), and store them in the `data` folder:

```bash
wget -P <path to this repository>/data https://rapids-single-cell-examples.s3.us-east-2.amazonaws.com/dsci_resting_nonzeropeaks.h5ad; \
wget -P <path to this repository>/data https://rapids-single-cell-examples.s3.us-east-2.amazonaws.com/dsci_resting_peaknames_nonzero.npy; \
wget -P <path to this repository>/data https://rapids-single-cell-examples.s3.us-east-2.amazonaws.com/dsci_resting_cell_metadata.csv
```

### Example Code

Follow this [Jupyter notebook](notebooks/dsci_bmmc_60k_gpu.ipynb) for RAPIDS analysis of this dataset. In order for the notebook to run, the files [rapids_scanpy_funcs.py](notebooks/rapids_scanpy_funcs.py) and [utils.py](notebooks/utils.py) need to be in the same folder as the notebook.

We provide a second notebook with the CPU version of this analysis [here](notebooks/dsci_bmmc_60k_cpu.ipynb).

### Acceleration

We report the runtime of these notebooks on various GCP instances below. All runtimes are given in seconds. Acceleration is given in parentheses. Benchmarking was performed on Dec 16, 2020.


| Step                         | CPU <br> e2-standard-32 <br> 32 vCPUs | GPU <br> n1-standard-16 <br> T4 16 GB GPU <br> (Acceleration)  | GPU <br> n1-highmem-8 <br> Tesla V100 16 GB GPU <br> (Acceleration) | GPU <br> a2-highgpu-1g <br> Tesla A100 40GB GPU <br> (Acceleration) |
|------------------------------|-------------------------|----------------------------|-------------------|-------------|
| PCA                          | 190                     | 137  (1.4x)                | 68   (2.8x)       | 52   (4x)   |
| KNN                          | 37                      | 17.9 (2.1x)                | 19.2 (2x)         | 4.6  (8x)   |
| UMAP                         | 33                      | 1.3  (25x)                 | 0.94 (35x)        | 0.68 (49x)  |
| Louvain clustering           | 7.5                     | 0.16 (47x)                 | 0.16 (47x)        | 0.10 (75x)  |
| Leiden clustering            | 11.4                    | 0.09 (127x)                | 0.08 (143x)       | 0.08 (143x) |
| t-SNE                        | 266                     | 3.6  (74x)                 | 2.1  (127x)       | 2.3  (116x) |
| Differential Peak Analysis   | 992                     | 23   (43x)                 | 17.6 (56x)        | 9.8  (101x) |
| End-to-end notebook run      | 1561                    | 182                        | 130               | 87          |
| Price ($/hr)                 | 1.073                   | 1.110                      | 2.953             | 4.00        |
| Total cost ($)               | 0.465                   | 0.063                      | 0.106             | 0.096       |

## Example 5: Visualizing Chromatin Accessibility in 5,000 PBMCs with RAPIDS and AtacWorks (Beta version)

<img align="left" width="240" height="200" src="https://github.com/avantikalal/rapids-single-cell-examples/blob/rilango/mem-fix/images/atacworks_notebook_img.png?raw=true">

We analyze single-cell ATAC-seq data from 5000 PBMC cells as in example 4. Additionally, we use cuDF to calculate and visualize cluster-specific chromatin accessibility in selected marker regions. Finally, we use a deep learning model trained with [AtacWorks](https://github.com/clara-parabricks/AtacWorks), to improve the accuracy of the chromatin accessibility track and call peaks in individual clusters.

### Example Data

The dataset was made publicly available by 10X Genomics. Use the following command to download the peak x cell count matrix and the fragment file for this dataset, and store both in the `data` folder:

```bash
wget -P <path to this repository>/data https://rapids-single-cell-examples.s3.us-east-2.amazonaws.com/5k_pbmcs_10X.sparse.h5ad
wget -P <path to this repository>/data https://rapids-single-cell-examples.s3.us-east-2.amazonaws.com/5k_pbmcs_10X_fragments.tsv.gz
wget -P <path to this repository>/data https://rapids-single-cell-examples.s3.us-east-2.amazonaws.com/5k_pbmcs_10X_fragments.tsv.gz.tbi
```

### Example Model

We use a pre-trained deep learning model to denoise the chromatin accessibility track and call peaks. This model can be downloaded into the `models` folder:
```bash
wget -P <path to this repository>/models https://api.ngc.nvidia.com/v2/models/nvidia/atac_bulk_lowcov_5m_50m/versions/0.3/files/models/model.pth.tar
```

### Example Code
Follow this [Jupyter notebook](notebooks/5k_pbmc_coverage_gpu.ipynb) for GPU analysis of this dataset. In order for the notebook to run, the files [utils.py](notebooks/utils.py), and [coverage.py](notebooks/coverage.py) need to be in the same folder as the notebook.



## Adapting these examples to another dataset

For our examples, we stored the count matrix in a sparse `.h5ad` format. To convert a different count matrix into this format, follow the instructions in [this notebook](notebooks/csv_to_h5ad.ipynb).
