from skimage import io
import numpy as np
import glob, os, re
from tqdm import tqdm
import sys


# load data
paths = []
labels = []
FRUIT_SET = []
LAST_SLASH = r'\/([^\/]*)\/[^\/]*$'
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
images = []

for p in tqdm(paths):
	try:
		img = io.imread(p)
	except ValueError:
		print( p )








