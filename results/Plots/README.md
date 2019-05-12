# Results Plot of Trained Models
Accuracy and loss plot of different models.

## Alexnet
* ***First attempt***  
  ```Optimizer```: RMSprop
  ```Batch Size``` : 128   
  ```epoch```: 200
  ```lr``` : 1e-6
* ***Second attempt***  
  ```Optimizer```: RMSprop
  ```Batch Size``` : 128   
  ```lr``` : 1e-5
* ***3rd attempt***   
  ```Optimizer```: Adam
  ```Batch Size``` : 128   
  ```lr``` : 1e-6
* ***4th attempt***   
  ```Optimizer```: Adam
  ```Batch Size``` : 128   
  ```lr``` : 1e-5
* ***5th attempt***   
  ```Optimizer```: Adam
  ```Batch Size``` : 128   
  ```lr``` : 1e-6
  ```no normalization layers```
* ***6th attempt***   
  ```Optimizer```: Adadelta
  ```Batch Size``` : 128   
  ```lr``` : 1e-3
* ***7th attempt***   
  ```Optimizer```: Adadelta
  ```Batch Size``` : 128   
  ```lr``` : 1e-2
  
## DenseNet121
* ***First attempt***   
  ```Batch Size``` : 32   
* ***Second attempt***   
  ```Batch Size``` : 64   
  ```lr``` : 0.01   
* ***3rd attempt***   
  ```Data``` : Full load   
  ```Batch Size``` : 128   
  ```lr``` : 0.001   
  ```validation_split``` : 0.3   
  ```test_acc``` : 0.65(false)   
* ***4th attempt***   
  ```Data``` : Full load   
  ```Batch Size``` : 512   
  ```lr``` : 0.001   
  ```validation_split``` : 0.3   
  ```test_acc``` : 0.94(false)      
* ***5th attempt***   
  ```Data``` : Full load   
  ```Batch Size``` : 512   
  ```lr``` : 1e-5   
  ```validation_split``` : 0.3   
  ```test_acc``` : 0.82396      
* ***6th attempt***   
  ```Data``` : Full load   
  ```Batch Size``` : 512   
  ```lr``` : 1e-5   
  ```dense block``` : 2 (original 4)   
  ```validation_split``` : 0.3   
  ```test_acc``` : 83.92%    
* ***7th attempt***   
  ```Data``` : Full load   
  ```Batch Size``` : 256   
  ```lr``` : 2e-5   
  ```dense block``` : 2 (original 4)   
  ```validation_split``` : 0.3   
  ```test_acc``` : 85.28%    
     
## VGG like 
* ***First attempt***   
  ```Batch Size``` : 128   
  ```lr``` : 1   
* ***Second attempt***   
  ```Batch Size``` : 64   
  ```lr``` : 0.1  
* ***3rd attempt***   
  ```Data``` : Full load   
  ```Batch Size``` : 128   
  ```lr``` : 0.01   
  ```validation_split``` : 0.2   
  ```test_acc``` :   0.84
* ***4th attempt***   
  ```Data``` : Full load   
  ```Batch Size``` : 64   
  ```lr``` : 5   
  ```validation_split``` : 0.2   
  ```test_acc``` :   0.85
 
