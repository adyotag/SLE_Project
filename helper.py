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
	images = np.zeros((len(paths), np.mul(RESIZE)))
	print('Importing images...')
	for i, p in enumerate(tqdm(paths)):
		img = io.imread(p)		# import image
		img = resize(img, RESIZE)	# resize image to reduce feature size
		images[i] = img.flatten()


	if savebackup:
		print('Saving Backup...')
		np.save(BACKUP+'X_30000.npy', images)
		np.save(BACKUP+'Y_ohk.npy', onehotkey)

	return [paths, onehotkey, FRUIT_SET, labels]



def poly(X, degree=1):
	returned_value = tc.zeros(X.shape[0], X.shape[1]*degree).float().to(CPU)
	for d in tc.arange(degree):
		returned_value[:, ((degree-1)*X.shape[1]):(degree*X.shape[1]) ] = tc.pow(X, degree)

	return returned_value


def LinearRegressionFit(X, y, epochs=1000):
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
