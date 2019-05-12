# Models
CNN models.
To run the python files, put ```balancedData_shuffled```, ```*.py```, ```and *_weights.h5``` under the same directory. To submit a training script ```*.py``` and train with GPUs, run the following command where ```train.sh``` will load the environment and execute the training script:
```
$ qsub -l gpus=1 -l gpu_c=3.5 train.sh
```
The result of training, validation and testing are images names ```accuracy.png``` and ```loss.png```.
   
## DenseNet
* Compile dense121_train.py
```
./dense121_train.py
```
  read images at ```'/projectnb/cs542sp/idc_classification/data/'```   
  process the Dataset and shuffle it, name as ```"balancedData_shuffled"``` then store it at ```'./balancedData_shuffled'```   
  Train the model and get the weights called ```"densenet_weights.h5"```.   
* Compile dense121_pred.py
```python
python dense121_pred.py
```
  load dataset called ```./balancedData_shuffled```   
  load weights for the model (```densenet_weights.h5```)   
  return the predicted result.
   
## VGG
  The file newvgg.py is our final model.
  1. read dataset at ```./balancedData_shuffled```
  2. output weights of the model: ```newvgg_weights.h5```
```python
python newvgg.py
```
   
## AlexNet
  The dataset is at ```./balancedData_shuffled```.
  The command:
```python
python alex_net.py
```
  could generate a model, and the command.
  
  The output weights is in this file: ```alexnet_model.h5```. It is a whole model for the training data, conatins optimized weights and model configuration. If you want to use it, import the ```load_model function``` from ```keras.model```.
