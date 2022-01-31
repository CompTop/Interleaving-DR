# Interleaving Dimension Reduction 
Repository for paper 'Topology-Preserving Dimensionality Reduction via Interleaving Optimization'. The images we used in paper are stored in folder `images` folder and codes in the `ipynb` folder.

## Installation
In order to re-run the experiments we have in the `ipynb` folder, you will need several packages installed first. 

### BATS.py
`bats` is used for general persistent homology computation, including greedy subsampling and computation flags. To install it, see the [installation page](https://bats-tda.readthedocs.io/en/latest/)(we suggest install from source files).

### torch_tda
`torch_tda` is used for optimzation on persistnet homology based on Pytorch, which supports auto differention. To install it, see the [installation page](https://torch-tda.readthedocs.io/en/latest/)(we suggest install from source files).

### Hera
Hera is used to compute bottleneck distance between two persistent diagrams, which has been proved to be much faster than Persim in our experiments. In order to install `hera_tda`, first install `boost` in your OS, then
```
git clone --recursive git@github.com:CompTop/pyhera.git
python setup.py install
```

## How to use
The main function for our dimension reduction method on a data set `X` with shape (n,p) is 
```Python
P, opt_info = bottleneck_proj_pursuit(X)
```
. It will return us a projection `P` with shape (p,2) and a dictionary  `opt_info` that stores optimzation information. You may also check a variety of parameters you can pass into the function in `PH_projection_pursuit.py`.

Next, in order to see the result in 2D plane, use 
```Python
X_PH = X @ P.T
plt.scatter(X_PH[:, 0],X_PH[:,1])
```