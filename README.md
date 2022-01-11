# Workload_GEN
This repo is used to generate workloads for **Learned Index** datasets. We recommand you use it with [SOSD](https://github.com/learnedsystems/sosd), a benchmark to compare (learned) index structures.

## Requirements
```
numpy
pandas
tqdm
matplotlib
```

## Usage
* `./download.sh` downloads and stores required data from the Internet
* `python generate.py` generates pre-defined workloads
* `python plot.py` visualizes the raw data and workloads

## Details
* The workload for each dataset contains *T* time periods (default of 10) and *k* lookups in every time period (default of 1e5), you can switch both of them to satisfy your requirements.
* The workload is developed based on `numpy.random`, we use five different distributions (i.e., zipf, normal, poisson, pareto, and binomial), you can select one or more of them in your workload, or you can use other suitable distributions.
* The workload is challenging compared to simply generated from raw data because you can design the workloads of following types by yourself:

    1. Same distribution in several time periods, but have different random hot points
    2. Same distribution in several time periods, but have different hot points obeying a certain distribution
    3. Different distributions in several time periods

## Getting Help
If you have any questions, please contact Lei Yang (yangop8@pku.edu.cn).