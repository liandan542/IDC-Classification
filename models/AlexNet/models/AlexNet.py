#import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""
print("loading packages;")
import pandas as pd
import numpy as np
from pickle import load, dump

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, AveragePooling2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import roc_curve, auc, roc_auc_score
from keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import Adam, SGD, Adadelta, RMSprop
from sklearn.metrics import confusion_matrix

print("building model;")
def alexnet_model(img_shape=(50, 50, 3), n_classes=2, l2_reg=0.,
    weights=None):

    # Initialize model
    alexnet = Sequential()

    # Layer 1
    alexnet.add(Conv2D(32, (5, 5), input_shape=img_shape))
    alexnet.add(BatchNormalization())
    
    # Layer 2
    alexnet.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    alexnet.add(Activation('relu'))    

    # Layer 3
    alexnet.add(Conv2D(32, (5, 5)))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    
    # Layer 4
    alexnet.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

    # Layer 5
    alexnet.add(Conv2D(64, (5, 5)))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    
    # Layer 6
    alexnet.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

    # Layer 7
    alexnet.add(Flatten())
    alexnet.add(Dropout(0.2))
    alexnet.add(Dense(64))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))

    # Layer 8
    alexnet.add(Dropout(0.2))
    alexnet.add(Dense(2))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    
    
    # Layer 9
    alexnet.add(Dense(n_classes))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('softmax'))
    

    if weights is not None:
        alexnet.load_weights(weights)

    return alexnet

def procData(data, lower_ind, upper_ind):
    x = []
    y = []
    y_vector = []
    for ind in range(lower_ind, upper_ind):
        path = data['path'][ind]
        label = data['label'][ind]
        image = data['matrix'][ind]
        shape = image.shape
        if shape == (50,50,3):
            x.append(image)
            if label == '1':
                y.append(np.asarray([0,1]))
                y_vector.append("1")
            else:
                y.append(np.asarray([1,0]))
                y_vector.append("0")
    
    return x, y, y_vector
    
print("loading data;")
file_path = "/projectnb/cs542sp/idc_classification/balancedData_shuffled"
all_data = pd.read_pickle(file_path)
all_data.head()

X_red, y_red, y_red_vec = procData(all_data, 0, 285048)
X_red, y_red, y_red_vec = np.asarray(X_red), np.asarray(y_red), np.asarray(y_red_vec)

X_test, y_test, y_test_vec= procData(all_data, 285048, 356310)
X_test, y_test, y_test_vec = np.asarray(X_test), np.asarray(y_test), np.asarray(y_test_vec)

X_red = X_red / 255.0
X_test = X_test / 255.0

# change the parameters for the model
epochs = 200
validation_split = .3
batch_size = 128
opt = RMSprop(lr=1e-6)

model = alexnet_model()

weight_path="{}_weights.best.hdf5".format('model')
earlystopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=0, mode='auto')
checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_weights_only=True)

print("fitting model;")
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
hist = model.fit(x = X_red, y = y_red, epochs=epochs, batch_size = batch_size, verbose = 2, callbacks = [earlystopping, checkpoint], validation_split=validation_split)
print("evaluating test data;")
test_loss,test_acc = model.evaluate(X_test, y_test, batch_size = batch_size)

print("predicting test data")
pred = model.predict(X_test, batch_size=batch_size)
prediction = []
tr = 0
fal = 0
print(len(pred))
print(len(y_test_vec))
for i in range(len(y_test_vec)):
    if pred[i][0] > pred[i][1]:
        y_pred = "0"
    else:
        y_pred = "1"
    
    prediction.append(y_pred)
    
    if y_pred == y_test_vec[i]:
        tr = tr+1
    else:
        fal = fal+1

acc = tr / (tr+fal)
cm = confusion_matrix(y_test_vec, prediction)
cm_acc = np.sum(cm.diagonal())/np.sum(cm)

print(cm)
print(cm_acc)

print("saving models and weights;")
model.save("../models/modified_alexnet_trainedmodel_GPU.h5") # saving the model 

print("saving h5 history;")
with open('trainHistoryOld', 'wb') as handle: # saving the history of the model trained for another 50 Epochs
    dump(hist.history, handle)

print("saving npy history")
history_acc_loss = {"loss": hist.history['loss'], "val_loss": hist.history['val_loss'], "acc": hist.history['acc'], "val_acc": hist.history['val_acc'], "final_test_loss_acc": [test_loss,test_acc], "confusion_matrix": cm, "cm_acc": [cm_acc, acc]}

np.save("history_acc_loss.npy", history_acc_loss)

print("ploting")
# Plot training & validation acc values
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('Test accuracy : ' + str(cm_acc) + ", " + str(acc))
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig("accuracy.png")
plt.close()

# Plot training & validation loss values
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Test loss : ' + str(test_loss))
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig("lost.png")
plt.close()

print("finished;")

print(model.summary())