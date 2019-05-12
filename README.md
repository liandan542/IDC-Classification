# IDC-Classification
The task of this project is to detect ```Invasive Ductal Carcinoma```(IDC) in slide images of Breast Cancer. The final product is a binary classifier telling whether an image patch is IDC positive or IDC negative. Our working directory on ```BU SCC``` is  ***/projectnb/cs542sp/idc_classification***.
   
* Group Name: **IDC**   

* Group Member:  
  + **Zeyu Fu** : zeyufu@bu.edu; 
  + **Yuanrong Liu** : yliu6680@bu.edu; 
  + **Yu Liu** : liuyu1@bu.edu   

## Directories in this Repo 

### Setups
The configuration files are located in this directory, including the environment installation, the GPU configurations etc,.

### Models
The Python scripts and Jupyter Notebook of our training and prediction are included in this directory. We use three models, ```AlexNet```, ```DenseNet``` and ```VGG```.

### Results
Under this directory, we have the records of accuracy and loss of every model we trained. The best results are shown on our presentation slides and final report.

## Dataset
Breast Histopathology image dataset is from [kaggle](https://www.kaggle.com/paultimothymooney/breast-histopathology-images), including 198,738 IDC(-) image patches and 78,786 IDC(+) image patches of 50*50 pixels. 
