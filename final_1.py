from helper import ExploreDataset
from helper import poly, LinearRegressionFit, LinearRegressionEval
from helper import LogisticRegressionFit, LogisticRegressionEval
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
import sys
import torch as tc


GPU = "cuda:0" 
CPU = "cpu"

#paths, onehotkey, categorical, FRUIT_SET, labels =  ExploreDataset(savebackup=True)

print('Loading Backup...')
images = tc.from_numpy( np.load('/data2/adyotagupta/school/X_30000.npy')  ).float().to(CPU)
onehotkey = tc.from_numpy( np.load('/data2/adyotagupta/school/Y_ohk.npy')  ).float().to(CPU)
categorical = tc.from_numpy( np.load('/data2/adyotagupta/school/Y_cat.npy')  ).int().to(CPU)
index = tc.arange(len(images)).float().to(CPU).unsqueeze(1)

# Shuffle data
data = tc.hstack([index, images])
data = data[tc.randperm(data.size()[0])]
shuffled_index = data[:,0].long()

X = data[:,1:]
y_ohk = onehotkey[shuffled_index]
y_cat = categorical[shuffled_index]
NUM_CATS = len(y_ohk[0]) 


# split into training, validation, and test
def split_dataset(arr, test=0.1, valid=0.1):
    LENGTH = len(X)
    D1, D2 = int(LENGTH*(1-valid-test)), int(LENGTH*(1-test))
    
    train_arr = arr[:D1]
    valid_arr = arr[D1:D2]
    test_arr = arr[D2:]
    return [train_arr, valid_arr, test_arr]


X_train, X_valid, X_test = split_dataset(X) 
y_ohk_train, y_ohk_valid, y_ohk_test = split_dataset(y_ohk) 
y_cat_train, y_cat_valid, y_cat_test = split_dataset(y_cat) 


X2_train = poly(X_train, 2)
X3_train = poly(X_train, 3)
X2_valid = poly(X_valid, 2)
X3_valid = poly(X_valid, 3)
X2_test = poly(X_test, 2)
X3_test = poly(X_test, 3)


## Linear Regression
print('Training Linear Regression models...\n')

print('Degree = 1')
DEG1 = LinearRegressionFit(X_train, y_ohk_train )
print('Train = {}'.format(LinearRegressionEval(X_train, y_ohk_train, DEG1[0]) ))
print('Validation = {}, Test = {}'.format(LinearRegressionEval(X_valid, y_ohk_valid, DEG1[0]), LinearRegressionEval(X_test, y_ohk_test, DEG1[0]) ))

print('Degree = 2')
DEG2 = LinearRegressionFit(X2_train, y_ohk_train )
print('Train = {}'.format(LinearRegressionEval(X2_train, y_ohk_train, DEG2[0]) ))
print('Validation = {}, Test = {}'.format(LinearRegressionEval(X2_valid, y_ohk_valid, DEG2[0]), LinearRegressionEval(X2_test, y_ohk_test, DEG2[0]) ))

print('Degree = 3')
DEG3 = LinearRegressionFit(X3_train, y_ohk_train )
print('Train = {}'.format(LinearRegressionEval(X3_train, y_ohk_train, DEG3[0]) ))
print('Validation = {}, Test = {}'.format(LinearRegressionEval(X3_valid, y_ohk_valid, DEG3[0]), LinearRegressionEval(X3_test, y_ohk_test, DEG3[0]) ))


# plot loss
plt.figure()

plt.plot(DEG1[1], label='Degree 1')
plt.plot(DEG2[1], label='Degree 2')
plt.plot(DEG3[1], label='Degree 3')

plt.title("Loss of Linear Regression Models over Epochs")
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.legend()
plt.savefig("LRLoss.png")


## Logistic Regression
print('\n\nTraining Logistic Regression models...\n')

print('Degree = 1')
DEG1 = LogisticRegressionFit(X_train, y_cat_train, NUM_CATS)
print('Train = {}'.format(LogisticRegressionEval(X_train, y_cat_train, DEG1[0]) ))
print('Validation = {}, Test = {}'.format(LogisticRegressionEval(X_valid, y_cat_valid, DEG1[0]),
						 LogisticRegressionEval(X_test, y_cat_test, DEG1[0]) ))

print('Degree = 2')
DEG2 = LogisticRegressionFit(X2_train, y_cat_train, NUM_CATS)
print('Train = {}'.format(LogisticRegressionEval(X2_train, y_cat_train, DEG2[0]) ))
print('Validation = {}, Test = {}'.format(LogisticRegressionEval(X2_valid, y_cat_valid, DEG2[0]),
						 LogisticRegressionEval(X2_test, y_cat_test, DEG2[0]) ))

print('Degree = 3')
DEG3 = LogisticRegressionFit(X3_train, y_cat_train, NUM_CATS)
print('Train = {}'.format(LogisticRegressionEval(X3_train, y_cat_train, DEG3[0]) ))
print('Validation = {}, Test = {}'.format(LogisticRegressionEval(X3_valid, y_cat_valid, DEG3[0]),
						 LogisticRegressionEval(X3_test, y_cat_test, DEG3[0]) ))

# plot loss
plt.figure()

plt.plot(DEG1[1], label='Degree 1')
plt.plot(DEG2[1], label='Degree 2')
plt.plot(DEG3[1], label='Degree 3')

plt.title("Loss of Logistic Regression Models over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Cross Entropy Loss")
plt.legend()
plt.savefig("LogRLoss.png")
