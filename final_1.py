from skimage import io
from matplotlib import pyplot as plt
from skimage.transform import resize
import numpy as np
import glob, os, re
from tqdm import tqdm
import sys
import torch as tc


GPU = "cuda:0" 
CPU = "cpu"

 

"""
# load data
paths = []
labels = []
FRUIT_SET = []
LAST_SLASH = r'\/([^\/]*)\/[^\/]*$'
print('Discovering files...')
for up in glob.glob( os.getcwd() + '/fruit_dataset/**/' ):	# for each fruit directory
	fruit = re.findall( LAST_SLASH, up )[0]
	FRUIT_SET.append( fruit )
	for p in glob.glob( up + '*' ):			# get image path of each fruit
		if fruit == 'Apple':				# but Apples have subdirectories
			for a in glob.glob( p + '/*' ):
				apple_type = re.findall( LAST_SLASH, a )[0]
				if apple_type == 'Total Number of Apples':	# Don't care about type of apple
					paths.append(a)
					labels.append( fruit )

		elif fruit == 'Kiwi':				# but Kiwi have subdirectories
			for k in glob.glob( p + '/*' ):
				apple_type = re.findall( LAST_SLASH, k )[0]
				if apple_type == 'Total Number of Kiwi fruit':	# Don't care about type of kiwi
					paths.append(k)
					labels.append( fruit )


		elif fruit == 'Guava':		# but Guava have subdirectories
			for g in glob.glob( p + '/*' ):
				apple_type = re.findall( LAST_SLASH, g )[0]
				if apple_type == 'guava total final':	# Don't care about type of guava
					paths.append(g)
					labels.append( fruit )

		else:
			paths.append(p)
			labels.append(fruit)


# generate category output and one-encoding outputs
categorical = np.zeros(len( labels  ), dtype=int)
onehotkey = np.zeros( (len(labels),len(FRUIT_SET)) ) 
for i, s in enumerate( labels  ):
	categorical[i] = np.argmax( np.asarray([ s == fs for fs in FRUIT_SET ], dtype=int )   )
	onehotkey[i,:] =  np.asarray([ s == fs for fs in FRUIT_SET ], dtype=int)


# Load images
BACKUP = '/data2/adyotagupta/school/'
RESIZE = (100,100,3)
images = np.zeros((len(paths), np.mul(RESIZE)))
print('Importing images...')
for i, p in enumerate(tqdm(paths)):
	img = io.imread(p)		# import image
	img = resize(img, RESIZE)	# resize image to reduce feature size
	images[i] = img.flatten()



print('Saving Backup...')
np.save(BACKUP+'X_30000.npy', images)
np.save(BACKUP+'Y_cat.npy', categorical)
np.save(BACKUP+'Y_ohk.npy', onehotkey)

"""

print('Loading Backup...')
images = tc.from_numpy( np.load('/data2/adyotagupta/school/X_30000.npy')  ).float().to(CPU)
onehotkey = tc.from_numpy( np.load('/data2/adyotagupta/school/Y_ohk.npy')  ).float().to(CPU)
categorical = tc.from_numpy( np.load('/data2/adyotagupta/school/Y_cat.npy')  ).float().to(CPU)
index = tc.arange(len(images)).float().to(CPU).unsqueeze(1)

# Shuffle data
data = tc.hstack([index, images])
data = data[tc.randperm(data.size()[0])]
shuffled_index = data[:,0].long()

X = data[:,1:]
y_cat = categorical[shuffled_index]
y_ohk = onehotkey[shuffled_index]


# split into training, validation, and test
def split_dataset(arr, test=0.1, valid=0.1):
    LENGTH = len(X)
    D1, D2 = int(LENGTH*(1-valid-test)), int(LENGTH*(1-test))
    
    train_arr = arr[:D1]
    valid_arr = arr[D1:D2]
    test_arr = arr[D2:]
    return [train_arr, valid_arr, test_arr]


X_train, X_valid, X_test = split_dataset(X) 
y_cat_train, y_cat_valid, y_cat_test = split_dataset(y_cat) 
y_ohk_train, y_ohk_valid, y_ohk_test = split_dataset(y_ohk) 



## Linear Regression
# adds squares, cubes, etc. of features for polynomial regression
def poly(X, degree=1):
	returned_value = tc.zeros(X.shape[0], X.shape[1]*degree).float().to(CPU)
	for d in tc.arange(degree):
		returned_value[:, ((degree-1)*X.shape[1]):(degree*X.shape[1]) ] = tc.pow(X, degree)

	return returned_value


def LinearRegressionFit(X, y, epochs=5000):
	X_GPU, y_GPU = X.to(GPU), y.to(GPU)
	INPUT = X.shape[1]
	try:  # one hot encoding
		OUTPUT = len(y[0])
	except TypeError: # categorical
		OUTPUT = 1
 
	model = tc.nn.Linear(INPUT, OUTPUT).to(GPU)
	loss_func = tc.nn.MSELoss()
	opt = tc.optim.SGD(model.parameters(), lr=0.001)	

	loss_counter = []


	pbar_epoch = tqdm(range(epochs))
	for epoch in pbar_epoch:
		y_hat = model(X_GPU.requires_grad_())
		loss = loss_func(y_hat, y_GPU)
		loss.backward()
		opt.step()
		opt.zero_grad()
		loss_counter.append(loss.item())
		pbar_epoch.set_description( "Loss: {}".format(loss.item()) )

	returned_value = [model.to(CPU), loss_counter]
	tc.cuda.empty_cache()
	return returned_value


def LinearRegressionEval(X, y, model):
	model_GPU, X_GPU, y_GPU = model.to(GPU), X.to(GPU), y.to(GPU)
	SAMPLES = len(y)
	y_hat = model_GPU(X_GPU)

	try:
		OUTPUT = len(y[0])
		g_hat = tc.zeros(SAMPLES, OUTPUT).byte().to(GPU)
		for i in range(SAMPLES):
			g_hat[i,tc.argmax(y_hat[i])] = 1
	except TypeError:
		OUTPUT = 1 
		g_hat = y_hat.byte() 

	correct = 0
	for i in range(SAMPLES):
		if (g_hat[i] == y_GPU[i].byte()).all():
			correct += 1.

	return correct/SAMPLES
	



X2_train = poly(X_train, 2)
X3_train = poly(X_train, 3)
X2_valid = poly(X_valid, 2)
X3_valid = poly(X_valid, 3)
X2_test = poly(X_test, 2)
X3_test = poly(X_test, 3)

print('Training Linear Regression models...')
DEG1 = LinearRegressionFit(X_train, y_ohk_train )
print('Train = {}'.format(LinearRegressionEval(X_train, y_ohk_train, DEG1[0]) ))
print('Validation = {}, Test = {}'.format(LinearRegressionEval(X_valid, y_ohk_valid, DEG1[0]), LinearRegressionEval(X_test, y_ohk_test, DEG1[0]) ))
DEG2 = LinearRegressionFit(X2_train, y_ohk_train )
print('Train = {}'.format(LinearRegressionEval(X2_train, y_ohk_train, DEG2[0]) ))
print('Validation = {}, Test = {}'.format(LinearRegressionEval(X2_valid, y_ohk_valid, DEG2[0]), LinearRegressionEval(X2_test, y_ohk_test, DEG2[0]) ))
DEG3 = LinearRegressionFit(X3_train, y_ohk_train )
print('Train = {}'.format(LinearRegressionEval(X3_train, y_ohk_train, DEG3[0]) ))
print('Validation = {}, Test = {}'.format(LinearRegressionEval(X3_valid, y_ohk_valid, DEG3[0]), LinearRegressionEval(X3_test, y_ohk_test, DEG3[0]) ))




plt.plot(DEG1[1])
plt.plot(DEG2[1])
plt.plot(DEG3[1])
plt.show()













