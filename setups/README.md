# Setups
This directory is the setups of how we training models on GPUs. 

## Environment
We create a new environment ```idc``` on **BU SCC**, which includes all the packages we need in training process. We can simply load the environment before we start working and do all the work in this environment. It is an efficient way of loading packages so that there will be no package missing and no version conflicts. 

## Usage
To install the environment, run the command:
```
$ sh init.sh
```
To activate the new environment called ```idc```, run:
```
$ source activate idc
```
To close the environment, run:
```
$ source deactivate
```
To submit a **Python** training script ```train.py``` and train with GPUs, run:
```
$ qsub -l gpus=1 -l gpu_c=3.5 train.sh
```


