'''
Rasterfairy code usage adapted from this example:
https://github.com/Quasimondo/RasterFairy/blob/master/
examples/Raster%20Fairy%20Demo%201.ipynb
'''

from gensim import models, matutils					# word2vec model loading
from sklearn.decomposition import IncrementalPCA	# inital reduction
from sklearn.manifold import TSNE 					# final reduction
import numpy as np 									# array handling
import os, warnings
import gensim
import matplotlib.pyplot as plt

model_filename =    'word2vec.model'		# model file to reduce
model_name = 		'music_word_2vec'							# name for exporting files

num_dimensions =     2				# final num dimensions (2D, 3D, etc)

run_init_reduction = True			# run an initial reduction with PCA?
init_dimensions =    300				# initial reduction before t-SNE

# use only most common words (helpful for big data sets)
only_most_common =   True
num_common = 		 50			# how many words to filter to? (max 50k)
tagged_pos = 		 False			# is our model tagged with parts-of-speech?

common_filename = 	 '50musiclabels.txt'


def normalize_list(vals):
	'''
	normalize a list of vectors to range of -1 to 1
	input: list of vectors
	output: normalized list
	'''
	min_val = float(min(vals))
	max_val = float(max(vals))
	output = []
	for val in vals:
		if val < 0:
			val = (val / min_val) * -1
		elif val > 0:
			val = val / max_val
		output.append(val)			# note if 0, stays the same :)
	return output


# ignore unicode warnings
warnings.filterwarnings('ignore', '.*Unicode.*')


# compute the model from GoogleNews-vectors-negative300 pretrained model
print ('computing model...') 
model =gensim.models.KeyedVectors.load_word2vec_format('<path to model>/magnatagatune/GoogleNews-vectors-negative300.bin',binary=True)
print ('- done')


# load model as numpy array
# if specified, keep only most common words
print ('converting model/words to numpy array...')
if only_most_common:
	print ('- loading ' + str(num_common) + ' most common words...')
	most_common = []
	with open(common_filename) as f:
		for i, line in enumerate(f):
			if i > num_common:
				break
			most_common.append(line.strip())

print ('- creating list of words/vectors for reduction...')
if only_most_common:
	print ('  - keeping only ' + str(num_common) + ' most common words')

vectors= []			# positions in vector space
labels = []			# keep track of words to label our data again later
for word in model.vocab:
	if only_most_common:
		if tagged_pos:
			parts = word.split('_')		# split _ for POS-tagged words
			w = parts[0].lower()
			p = parts[1]
			if w in most_common:
				word = w + '_' + p
				try:
					vectors.append(model[word])
					labels.append(word)
				except:
					pass
		else:
			if word in most_common:
				try:
					vectors.append(model[word])
					labels.append(word)
				except:
					pass
	else:
		vectors.append(model[word])
		labels.append(word)
print ('- found ' + str(len(labels)) + ' entities x ' + str(len(vectors[0])) + ' dimensions')


# convert both lists into numpy vectors for reduction
vectors = np.asarray(vectors)
labels =  np.asarray(labels)
print ('- done')


# if specified, reduce using IncrementalPCA first (down 
# to a smaller number of dimensions before the final reduction)
if run_init_reduction:
	print ('reducing to ' + str(init_dimensions) + 'D using IncrementalPCA...')
	ipca = IncrementalPCA(n_components=init_dimensions)
	vectors = ipca.fit_transform(vectors)
	print ('- done')

	# save reduced vector space to file
	print ('- saving as csv...')
	with open(''+model_name + '-' + str(init_dimensions) + 'D.csv', 'w') as f:
		for i in range(len(labels)):
			f.write(labels[i] + ',' + ','.join(map(str, vectors[i])) + '\n')


# reduce using t-SNE
print ('reducing to ' + str(num_dimensions) + 'D using t-SNE...')
print ('- may take a really, really (really) long time :)')
vectors = np.asarray(vectors)
tsne = TSNE(n_components=num_dimensions, random_state=0)
vectors = tsne.fit_transform(vectors)
print ('- done')


# save reduced vector space to file
print ('saving as csv...')
x_vals = [ v[0] for v in vectors ]
y_vals = [ v[1] for v in vectors ]
#z_vals = [ v[2] for v in vectors ]
#w_vals = [ v[3] for v in vectors ]

with open('' + model_name + '-' + str(num_dimensions) + 'D.csv', 'w') as f:
	for i in range(len(labels)):
         label = labels[i]
         x = x_vals[i]
         y = y_vals[i]
#         z = z_vals[i]
#         w = w_vals[i]
         f.write(label + ',' + str(x) + ',' + str(y) 
#         +','+str(z) + ',' + str(w) 
         + '\n')
print ('- done')


# normalize values -1 to 1, save to file
print ('normalizing position values...')
x_vals = normalize_list(x_vals)
y_vals = normalize_list(y_vals)
#z_vals = normalize_list(z_vals)
#w_vals = normalize_list(w_vals)
print ('- saving as csv...')
with open('' + model_name + '-' + str(num_dimensions) + 'D-NORMALIZED.csv', 'w') as f:
    for i in range(len(labels)):
         label = labels[i]
         x = x_vals[i]
         y = y_vals[i]
#         z = z_vals[i]
#         w = w_vals[i]
         f.write(label + ',' + str(x) + ',' + str(y) 
#         +','+str(z) + ',' + str(w) 
         + '\n')
print ('- done')



def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')
    plt.savefig(filename)



# Finally plotting and saving the fig 
plot_with_labels(vectors, labels)
