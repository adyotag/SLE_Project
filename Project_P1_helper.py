from skimage import io
from skimage.transform import resize
from matplotlib import pyplot as plt
import numpy as np	
import glob, os, re
from tqdm import tqdm
import torch as tc


GPU = "cuda:0" 
CPU = "cpu"

def ExploreDataset(savebackup = False):
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
	images = np.zeros((len(paths), np.prod(RESIZE)))
	print('Importing images...')
	for i, p in enumerate(tqdm(paths)):
		img = io.imread(p)		# import image
		img = resize(img, RESIZE)	# resize image to reduce feature size
		images[i] = img.flatten()


	if savebackup:
		print('Saving Backup...')
		np.save(BACKUP+'X_30000.npy', images)
		np.save(BACKUP+'Y_ohk.npy', onehotkey)
		np.save(BACKUP+'Y_cat.npy', categorical)

	return [paths, onehotkey, categorical, FRUIT_SET, labels]



def poly(X, degree=1):
	returned_value = tc.zeros(X.shape[0], X.shape[1]*degree).float().to(CPU)
	for d in tc.arange(degree):
		returned_value[:, ((degree-1)*X.shape[1]):(degree*X.shape[1]) ] = tc.pow(X, degree)

	return returned_value


def LinearRegressionFit(X, y, epochs=1000):
	X_GPU, y_GPU = X.to(GPU), y.to(GPU)
	INPUT, OUTPUT = X.shape[1], len(y[0])
 
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
	SAMPLES, OUTPUT = len(y), len(y[0])
	y_hat = model_GPU(X_GPU)

	g_hat = tc.zeros(SAMPLES, OUTPUT).byte().to(GPU)
	for i in range(SAMPLES):
		g_hat[i,tc.argmax(y_hat[i])] = 1

	correct = 0
	for i in range(SAMPLES):
		if (g_hat[i] == y_GPU[i].byte()).all():
			correct += 1.

	tc.cuda.empty_cache()
	return correct/SAMPLES



def LogisticRegressionFit(X, y, NUM_CATS, epochs=1000 ):
	X_GPU, y_GPU = X.to(GPU), y.to(GPU)
	INPUT, OUTPUT = X.shape[1], NUM_CATS
 
	layers = [tc.nn.Linear(INPUT, OUTPUT), tc.nn.Sigmoid()]
	model = tc.nn.Sequential(*layers).to(GPU)


	loss_func = tc.nn.CrossEntropyLoss()
	opt = tc.optim.SGD(model.parameters(), lr=.1)	

	loss_counter = []

	pbar_epoch = tqdm(range(epochs))
	for epoch in pbar_epoch:
		y_hat = model(X_GPU.requires_grad_())
		loss = loss_func(y_hat, y_GPU.long())
		loss.backward()
		opt.step()
		opt.zero_grad()
		loss_counter.append(loss.item())
		pbar_epoch.set_description( "Loss: {}".format(loss.item()) )

	returned_value = [model.to(CPU), loss_counter]
	tc.cuda.empty_cache()
	return returned_value


def LogisticRegressionEval(X, y, model):
	model_GPU, X_GPU, y_GPU = model.to(GPU), X.to(GPU), y.to(GPU)
	SAMPLES = len(y)
	y_hat = model_GPU(X_GPU)

	g_hat = tc.zeros(SAMPLES).byte().to(GPU)
	for i in range(SAMPLES):
		g_hat[i] = tc.argmax(y_hat[i])


	correct = 0
	for i in range(SAMPLES):
		if g_hat[i] == y_GPU[i].byte():
			correct += 1.

	tc.cuda.empty_cache()
	return correct/SAMPLES


def PlotANN(h, i):
	print('Saving figures for Model {}...'.format(i))
	plt.figure()
	plt.plot(h['accuracy'], label='train')
	plt.plot(h['val_accuracy'], label='validation')
	plt.title('Accuracy Plot for Model {}'.format(i))
	plt.ylabel('Accuracy')
	plt.xlabel('Epochs')
	plt.legend()
	plt.savefig('model{}_acc.png'.format(i))

	plt.figure()
	plt.plot(h['loss'], label='train')
	plt.plot(h['val_loss'], label='validation')
	plt.title('Loss Plot for Model {}'.format(i))
	plt.ylabel('Loss')
	plt.xlabel('Epochs')
	plt.legend()
	plt.savefig('model{}_loss.png'.format(i))















#
