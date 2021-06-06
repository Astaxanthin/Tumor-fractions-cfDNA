# Structural Referen-free Deconvolution and Bayesian Inference (sRFDBI)

Official MATLAB implementation of structural reference-free deconvolution and Bayesian inference (sRFDBI) algorithm.


## Prerequisites

MATLAB R2018b

## Usage
```
run main.m
```
*Initialization.m* is a parameter configuration file.

Options in the parameter configuration file:
+ *file_dir* : The directory of data. 
+ *param* : The parameters for the sRFDBI algorithm, including $p$ in $l_{2,p}$ norm 
Coefficient of structural penalty $\lambda$ 
The number of cancer methylation patterns 
The number of healthy methylation patterns 
Convergence threshold $\varepsilon$ 
Maximum iterations $T$ and prior. 
If *simulation_data* is used, *cnv* is available to choose the probability of CNV event.
If *real_data* is used, *top-k* is available to set the number of methylation markers.

## File instruction

The 'data' directory contains two exemplary datasets: simulation dataset and real dataset.

+ *simulation_dataset*

  The simulation data contains three datasets with the CNV event probabilities of 10%, 30% and 50%. Each dataset consists of 3000 training (500 for each category) and 1800 test (300 for each category) samples. 

  *train_data* is a $(K_s+1) \times N_1$ matrix. The first $K_s$ rows represent methylation data ( $N_1$ samples, each with $K_s$ dimensional methylation levels, i.e. $\beta$ value). The last row suggests the category of each training sample. Similarly, *test_data* is a $(K_s+1) \times N_2$ matrix. *train_theta* and *test_theta* denote the simulated tumor fraction of training and test samples.

+ *real_dataset*
  
  *real_data* is a $(K_r+1) \times N$ matrix. The first $K_r$ rows represent methylation data ( $N$ samples, each with $K_r$ dimensional methylation levels, i.e. $\beta$ value). The last row suggests the category of each sample.

The final results for each dataset contains two rows: the first row is cancer diagnostic results and the second row indicates tumor fraction prediction.

## References
If you find this work or code useful, please cite this study. The citation will be updated soon.

