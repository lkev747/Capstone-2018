# Gensim Doc2Vec Tutorial - Musings of a Panda
# http://linanqiu.github.io/2015/10/07/word2vec-sentiment/

# Set Up and Import Statements
from gensim import utils
from gensim.models.deprecated.doc2vec import LabeledSentence
from gensim.models import Doc2Vec

import numpy
from random import shuffle
from sklearn.linear_model import LogisticRegression


model = Doc2Vec.load('./toxic.d2v')

# Word Embedding Testing 

if ('y' == input('Test Word Embeddings? (y/n): ')):
	tmp = input('Enter a word to test (0 to quit): ')
	while(tmp != '0'):
		print(model.most_similar(tmp))
		tmp = input('Enter a word to test (0 to quit): ')


# Sentiment Classification 
	# First we use the document vectors to train a classifier
	# There are 159545 total entries (16199 toxic, 143346 non_toxic)


train_arrays = numpy.zeros((32398, 300))
train_labels = numpy.zeros(32398)

for i in range(16199):
	prefix_train_tox = 'TRAIN_TOXIC_' + str(i)
	prefix_train_non = 'TRAIN_NONTOXIC_' + str(i)
	
	train_arrays[i] = model[prefix_train_tox]
	train_arrays[16199 + i] = model[prefix_train_non]

	train_labels[i] = 1
	train_labels[16199 + i] = 0

classifier = LogisticRegression()
classifier.fit(train_arrays, train_labels)

if ('y' == input('Test classifier? (y/n): ')):
	tmp = input('Enter a phrase (0 to quit): ')
	while(tmp != '0'):
		vec = model.infer_vector(utils.simple_preprocess(doc = tmp, deacc = True))
		prob = classifier.predict_proba(vec.reshape(1, -1))
		
		if (prob[0][0] > prob[0][1]):
			print('Non-toxic')
			print('Confidence: ', prob[0][0])
		else:
			print('Toxic')
			print('Confidence: ', prob[0][1])
		
		tmp = input('Enter a phrase (0 to quit): ')