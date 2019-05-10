# Results Plot of Trained Models
Accuracy and loss plot of different models.
   
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
