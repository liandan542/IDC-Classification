# DenseNet121
A editted version of DenseNet121, reduced the dense blocks. The jupyter notebooks not fully working because we have rather large dataset and there is limitation of using CPUs. We then turn to using GPUs by submitting shell scripts.   
   
* ```DenseNet.ipynb```   
Some training experiments we do in **Jupyter Notebbok**, only small load of data feeded.   
   
* ```dense121_train.py```   
A Python script to train a editted version of DenseNet121 with full load data, run by submit to GPU by shell scripts. See details in [Usage](https://github.com/liandan542/IDC-Classification/tree/master/setups#usage).   
   
* ```densenet_weights.h5```    
Trained model weights.    
   
* ```dense121_pred.py```   
A prediction function to compute the regular accuracy and balance accuracy, by loading a model weight file.   
   
