from helper import poly, LinearRegressionFit, LinearRegressionEval
from helper import LogisticRegressionFit, LogisticRegressionEval
from helper import ExploreDataset, PlotANN
from matplotlib import pyplot as plt
import numpy as np
import torch as tc
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
import pickle
import sys

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


X_trains = [X_train, X2_train, X3_train]
X_valids = [X_valid, X2_valid, X3_valid]
X_tests = [X_test, X2_test, X3_test]


"""
## Linear Regression
print('Training Linear Regression models...\n')
plt.figure()

for i in range(3):
	DEGREE = i+1
	print('Degree = {}'.format(DEGREE))
	DEG = LinearRegressionFit(X_trains[i], y_ohk_train )
	print('Train = {}'.format(LinearRegressionEval(X_trains[i], y_ohk_train, DEG[0]) ))
	print('Validation = {}, Test = {}'.format(LinearRegressionEval(X_valids[i], y_ohk_valid, DEG[0]), LinearRegressionEval(X_tests[i], y_ohk_test, DEG[0]) ))

	# plot loss
	plt.plot(DEG[1], label='Degree {}'.format(DEGREE))

	tc.cuda.empty_cache()

plt.title("Loss of Linear Regression Models over Epochs")
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.legend()
plt.savefig("LRLoss.png")


## Logistic Regression
print('\n\nTraining Logistic Regression models...\n')
plt.figure()

for i in range(3):
	DEGREE = i+1
	print('Degree = {}'.format(DEGREE))
	DEG = LogisticRegressionFit(X_trains[i], y_cat_train, NUM_CATS )
	print('Train = {}'.format(LogisticRegressionEval(X_trains[i], y_cat_train, DEG[0]) ))
	print('Validation = {}, Test = {}'.format(LogisticRegressionEval(X_valids[i], y_cat_valid, DEG[0]), LogisticRegressionEval(X_tests[i], y_cat_test, DEG[0]) ))

	# plot loss
	plt.plot(DEG[1], label='Degree {}'.format(DEGREE))

	tc.cuda.empty_cache()


plt.title("Loss of Logistic Regression Models over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Cross Entropy Loss")
plt.legend()
plt.savefig("LogRLoss.png")
"""


## Artificial Neural Network

INPUT_SHAPE_T = tuple([len(X_train),100,100,3])
INPUT_SHAPE_V = tuple([len(X_valid),100,100,3])
INPUT_SHAPE_TEST = tuple([len(X_test),100,100,3])

X_train = X_train.cpu().detach().numpy().reshape(INPUT_SHAPE_T)
y_ohk_train = y_ohk_train.cpu().detach().numpy()
X_valid = X_valid.cpu().detach().numpy().reshape(INPUT_SHAPE_V) 
y_ohk_valid = y_ohk_valid.cpu().detach().numpy()
X_test = X_test.cpu().detach().numpy().reshape(INPUT_SHAPE_TEST)
y_ohk_test = y_ohk_test.cpu().detach().numpy()

print('Training Artificial NN models...\n')

# model 1
print('Model 1...\n')
model1 = Sequential()
layers = [Conv2D(16, 32, activation='sigmoid', input_shape=(100,100,3) ),
	  Conv2D(8, 16, activation='sigmoid' ),
	  Conv2D(4, 8, activation='sigmoid' ),
	  Flatten(),
	  Dense(1000, activation='relu'),
	  Dense(NUM_CATS, activation='softmax' )]

[model1.add(l) for l in layers]

model1.compile(loss=keras.losses.categorical_crossentropy,
		optimizer=keras.optimizers.SGD(lr=0.01),
		metrics=['accuracy'])

history1 = model1.fit(X_train, y_ohk_train,
          	batch_size=256,
	        epochs=150,
        	verbose=1,
	        validation_data=(X_valid, y_ohk_valid),)

PlotANN(history1.history, 1)

f = open('/data2/adyotagupta/school/history1.pkl','wb')
pickle.dump(history1.history, f)
f.close()

print('Test Accuracy = {}'.format(model1.evaluate(X_test, y_ohk_test)))

print('Saving model...\n')
model1.save('/data2/adyotagupta/school/model1.mod')

#Saving figures for Model 1...
#Epoch 150/150
#139/139 [==============================] - 8s 57ms/step - loss: 0.1472 - accuracy: 0.9604 - val_loss: 0.3086 - val_accuracy: 0.9005
#139/139 [==============================] - 1s 7ms/step - loss: 0.2904 - accuracy: 0.9048
#Test Accuracy = [0.29041942954063416, 0.904751181602478]



# Model 2
print('Model 2 ...\n')
model2 = Sequential()
layers = [Conv2D(16, 32, activation='sigmoid', input_shape=(100,100,3) ),
	  Conv2D(8, 16, activation='sigmoid' ),
	  Conv2D(4, 8, activation='sigmoid' ),
	  Flatten(),
	  Dense(1000, activation='sigmoid'),
	  Dropout(0.1),
	  Dense(NUM_CATS, activation='softmax' )]

[model2.add(l) for l in layers]

model2.compile(loss=keras.losses.categorical_crossentropy,
		optimizer=keras.optimizers.SGD(lr=0.01),
		metrics=['accuracy'])

history2 = model2.fit(X_train, y_ohk_train,
          	batch_size=256,
	        epochs=150,
        	verbose=1,
	        validation_data=(X_valid, y_ohk_valid),)


PlotANN(history2.history, 2)

f = open('/data2/adyotagupta/school/history2.pkl','wb')
pickle.dump(history2.history, f)
f.close()


print('Test Accuracy = {}'.format(model2.evaluate(X_test, y_ohk_test)))
print('Saving model...\n')
model2.save('/data2/adyotagupta/school/model2.mod')


#Saving figures for Model 2...
#Epoch 150/150
#139/139 [==============================] - 8s 56ms/step - loss: 0.5186 - accuracy: 0.8279 - val_loss: 0.7289 - val_accuracy: 0.7447
#139/139 [==============================] - 1s 4ms/step - loss: 0.7128 - accuracy: 0.7465
#Test Accuracy = [0.7128385305404663, 0.7464535236358643]



