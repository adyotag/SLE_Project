from Project_P1_helper import poly, LinearRegressionFit, LinearRegressionEval
from Project_P1_helper import LogisticRegressionFit, LogisticRegressionEval
from Project_P1_helper import ExploreDataset, PlotANN
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from keras.models import Sequential
from matplotlib import pyplot as plt
import pickle, sys
import numpy as np
import torch as tc
import keras

# pytorch settings
GPU = "cuda:0" 
CPU = "cpu"

# Explore, Process and save Processed Dataset
SAVE_PATH = '/data2/adyotagupta/school/'	# you will need to change this to ensure it runs on your system
paths, onehotkey, categorical, FRUIT_SET, labels =  ExploreDataset(SAVE_PATH)

print('Loading Backup...')
images = tc.from_numpy( np.load(SAVE_PATH+'X_30000.npy')  ).float().to(CPU)  # load from save file and make torch tensor
onehotkey = tc.from_numpy( np.load(SAVE_PATH+'Y_ohk.npy')  ).float().to(CPU)
categorical = tc.from_numpy( np.load(SAVE_PATH+'Y_cat.npy')  ).int().to(CPU)
index = tc.arange(len(images)).float().to(CPU).unsqueeze(1)

# Shuffle data
data = tc.hstack([index, images])
data = data[tc.randperm(data.size()[0])]
shuffled_index = data[:,0].long()

X = data[:,1:]	# shuffled data, X and y
y_ohk = onehotkey[shuffled_index]
y_cat = categorical[shuffled_index]
NUM_CATS = len(y_ohk[0]) # number of categories, 15 


# split into training, validation, and test
# default is 90% of data goes to training and validation
def split_dataset(arr, test=0.1, valid=0.1):
    LENGTH = len(X)
    D1, D2 = int(LENGTH*(1-valid-test)), int(LENGTH*(1-test))
    
    train_arr = arr[:D1]
    valid_arr = arr[D1:D2]
    test_arr = arr[D2:]
    return [train_arr, valid_arr, test_arr]


# split data into different sets
X_train, X_valid, X_test = split_dataset(X) 
y_ohk_train, y_ohk_valid, y_ohk_test = split_dataset(y_ohk) 
y_cat_train, y_cat_valid, y_cat_test = split_dataset(y_cat) 


# create inputs for polynomial and logistic regression
X2_train = poly(X_train, 2)
X3_train = poly(X_train, 3)
X2_valid = poly(X_valid, 2)
X3_valid = poly(X_valid, 3)
X2_test = poly(X_test, 2)
X3_test = poly(X_test, 3)


# degree=1, degree=2, degree=3
X_trains = [X_train, X2_train, X3_train]
X_valids = [X_valid, X2_valid, X3_valid]
X_tests = [X_test, X2_test, X3_test]


## Polynomial Regression
print('Training Polynomial Regression models...\n')
plt.figure()

for i in range(3):
	DEGREE = i+1
	print('Degree = {}'.format(DEGREE))
	DEG = LinearRegressionFit(X_trains[i], y_ohk_train ) # fit using linear regression algorithm
	# Evaluate Training, Validation, and Testing data sets
	print('Train = {}'.format(LinearRegressionEval(X_trains[i], y_ohk_train, DEG[0]) ))
	print('Validation = {}, Test = {}'.format(LinearRegressionEval(X_valids[i], y_ohk_valid, DEG[0]), LinearRegressionEval(X_tests[i], y_ohk_test, DEG[0]) ))

	# plot loss
	plt.plot(DEG[1], label='Degree {}'.format(DEGREE))

	tc.cuda.empty_cache()

# Save plot
plt.title("Loss of Polynomial Regression Models over Epochs")
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.legend()
plt.savefig("PRLoss.png")


## Logistic Regression
print('\n\nTraining Logistic Regression models...\n')
plt.figure()

for i in range(3):
	DEGREE = i+1
	print('Degree = {}'.format(DEGREE))
	DEG = LogisticRegressionFit(X_trains[i], y_cat_train, NUM_CATS ) # fit using logistic regression
	# Evaluate Training, Validation, and Testing data sets
	print('Train = {}'.format(LogisticRegressionEval(X_trains[i], y_cat_train, DEG[0]) )) 
	print('Validation = {}, Test = {}'.format(LogisticRegressionEval(X_valids[i], y_cat_valid, DEG[0]), LogisticRegressionEval(X_tests[i], y_cat_test, DEG[0]) ))

	# plot loss
	plt.plot(DEG[1], label='Degree {}'.format(DEGREE))

	tc.cuda.empty_cache()

# Save plot
plt.title("Loss of Logistic Regression Models over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Cross Entropy Loss")
plt.legend()
plt.savefig("LogRLoss.png")



## Artificial Neural Network
INPUT_SHAPE_T = tuple([len(X_train),100,100,3])
INPUT_SHAPE_V = tuple([len(X_valid),100,100,3])
INPUT_SHAPE_TEST = tuple([len(X_test),100,100,3])

# convert back to numpy arrays in preparation for keras
X_train = X_train.cpu().detach().numpy().reshape(INPUT_SHAPE_T)
y_ohk_train = y_ohk_train.cpu().detach().numpy()
X_valid = X_valid.cpu().detach().numpy().reshape(INPUT_SHAPE_V) 
y_ohk_valid = y_ohk_valid.cpu().detach().numpy()
X_test = X_test.cpu().detach().numpy().reshape(INPUT_SHAPE_TEST)
y_ohk_test = y_ohk_test.cpu().detach().numpy()

print('Training Artificial NN models...\n')

# Model 1 -- no dropout
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

f = open(SAVE_PATH+'history1.pkl','wb')
pickle.dump(history1.history, f)
f.close()

print('Test Accuracy = {}'.format(model1.evaluate(X_test, y_ohk_test)))
print('Saving model...\n')
model1.save(SAVE_PATH+'model1.mod')


# Model 2 -- dropout 5%
print('Model 2 ...\n')
model2 = Sequential()
layers = [Conv2D(16, 32, activation='sigmoid', input_shape=(100,100,3) ),
	  Conv2D(8, 16, activation='sigmoid' ),
	  Conv2D(4, 8, activation='sigmoid' ),
	  Flatten(),
	  Dropout(0.05),
	  Dense(1000, activation='sigmoid'),
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

f = open(SAVE_PATH+'history2.pkl','wb')
pickle.dump(history2.history, f)
f.close()


print('Test Accuracy = {}'.format(model2.evaluate(X_test, y_ohk_test)))
print('Saving model...\n')
model2.save(SAVE_PATH+'model2.mod')


# Model 3 -- dropout 10%
print('Model 3 ...\n')
model3 = Sequential()
layers = [Conv2D(16, 32, activation='sigmoid', input_shape=(100,100,3) ),
	  Conv2D(8, 16, activation='sigmoid' ),
	  Conv2D(4, 8, activation='sigmoid' ),
	  Flatten(),
	  Dropout(0.1),
	  Dense(1000, activation='sigmoid'),
	  Dense(NUM_CATS, activation='softmax' )]

[model3.add(l) for l in layers]

model3.compile(loss=keras.losses.categorical_crossentropy,
		optimizer=keras.optimizers.SGD(lr=0.01),
		metrics=['accuracy'])

history3 = model3.fit(X_train, y_ohk_train,
          	batch_size=256,
	        epochs=150,
        	verbose=1,
	        validation_data=(X_valid, y_ohk_valid),)


PlotANN(history3.history, 3)

f = open(SAVE_PATH+'history3.pkl','wb')
pickle.dump(history3.history, f)
f.close()


print('Test Accuracy = {}'.format(model3.evaluate(X_test, y_ohk_test)))
print('Saving model...\n')
model3.save(SAVE_PATH+'model3.mod')


# Model 4 -- dropout 2.5%
print('Model 4 ...\n')
model4 = Sequential()
layers = [Conv2D(16, 32, activation='sigmoid', input_shape=(100,100,3) ),
	  Conv2D(8, 16, activation='sigmoid' ),
	  Conv2D(4, 8, activation='sigmoid' ),
	  Flatten(),
	  Dropout(0.025),
	  Dense(1000, activation='sigmoid'),
	  Dense(NUM_CATS, activation='softmax' )]

[model4.add(l) for l in layers]

model4.compile(loss=keras.losses.categorical_crossentropy,
		optimizer=keras.optimizers.SGD(lr=0.01),
		metrics=['accuracy'])

history4 = model4.fit(X_train, y_ohk_train,
          	batch_size=256,
	        epochs=150,
        	verbose=1,
	        validation_data=(X_valid, y_ohk_valid),)


PlotANN(history4.history, 4)

f = open(SAVE_PATH+'history4.pkl','wb')
pickle.dump(history4.history, f)
f.close()


print('Test Accuracy = {}'.format(model4.evaluate(X_test, y_ohk_test)))
print('Saving model...\n')
model4.save(SAVE_PATH+'model4.mod')







#
