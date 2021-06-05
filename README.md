# Structural Referen-free Deconvolution and Bayesian Inference (sRFDBI)

Official MATLAB implementation of structural reference-free deconvolution and Bayesian inference (sRFDBI) algorithm.


## Prerequisites

MATLAB R2018b

## Usage
```
run main.m
```
Initialization.m is a parameter configuration file.

Options in the parameter configuration file:
+ *train_data_dir*: the address of training data. 
+ *test_data_dir*: the address of test data.
+ *param*: the parameters for the sRFDBI algorithm, including $p$ in $l_{2,p}$ norm, coefficient of structural penalty $\lambda$, the number of cancer methylation patterns, the number of healthy methylation patterns, convergence threshold $\varepsilon$, maximum iterations $T$ and prior.

## File instruction

The 'data' directory contains two exemplary datasets: simulation dataset and real dataset.

+ *simulation_dataset*: *train_data* is a $(K+1) \times N_1$ matrix. The first $K$ rows represent methyaltion data ( $N_1$ samples, each with K dimentional methylation levels, i.e. $\beta$ value). The last row suggests the category of each training sample. Similarly, *test_data* is a $(K+1) \times N_2$ matrix. *train_theta* and *test_theta* denote the simulated tumor fraction of training and test samples.
+ *real_dataset*: *train_data* and *test_data* are generated by randomly splitting the original dataset with a ratio of $1:1$.

The final results for each dataset contains two rows: the first row is cancer diagnostic results and the second row indicates tumor fraction prediction.

## References
If you find this work or code useful, please cite:

