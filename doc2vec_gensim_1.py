# Gensim Doc2Vec Tutorial - Musings of a Panda
# http://linanqiu.github.io/2015/10/07/word2vec-sentiment/

# Set Up and Import Statements
from gensim import utils
#from gensim.models.deprecated.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
from gensim import models
import gensim

from gensim.models.doc2vec import TaggedDocument

import numpy
from random import shuffle
#from sklearn.linear_model import LogisticRegression

import os
import glob
import smart_open
import multiprocessing

assert gensim.models.doc2vec.FAST_VERSION > -1
cores = multiprocessing.cpu_count()

# Input Format
#	Clean the data by converting everything to lowercase and removing punctuation
#	Each document should be on one line, separated by new lines!!!
#	Sentences should be formatted in this way:
#		[['word1', 'word2', 'word3', 'lastword'], ['label1']]
#		LabeledSentence is a tidier way to do that
#	We write our own class to handle multiple files/documents
'''
class LabeledLineSentence(object):
	def __init__(self, sources):
		self.sources = sources

		flipped = {}

		for key, value in sources.items():
			if value not in flipped:
				flipped[value] = [key]
			else:
				raise Exception('Non-unique prefix encountered')

	def __iter__(self):
		for source, prefix in self.sources.items():
			with utils.smart_open(source) as fin:
				for item_no, line in enumerate(fin):
					yield LabeledSentence(utils.to_unicode(line).split(), [prefix + '_%s' % item_no])

	def to_array(self):
		self.sentences = []
		for source, prefix in self.sources.items():
			with utils.smart_open(source) as fin:
				for item_no, line in enumerate(fin):
					self.sentences.append(LabeledSentence(utils.to_unicode(line).split(), [prefix + '_%s' % item_no]))
		return self.sentences
    
	def sentences_perm(self):
		shuffle(self.sentences)
		return self.sentences
'''

def read_corpus(fname, tag):
	with smart_open.smart_open(fname) as f:
		for i, line in enumerate(f):
			tags = tag + "_" + str(i)
			yield TaggedDocument(utils.simple_preprocess(line), [tags])


def read_corpus_folder(folders, tag):
	ID = 0
	for fol in folders:
		txt_files = glob.glob(os.path.join(fol, '*.txt'))
		for file in txt_files:
			tags = tag + "_" + str(ID)
			ID += 1
			with smart_open.smart_open(file) as f:
				for i, line in enumerate(f):
					yield TaggedDocument(utils.simple_preprocess(line), [tags])


# folders = ['train/pos', 'train/neg', 'test/pos', 'test/neg', 'train/unsup']


sentences = list(read_corpus("toxic.txt", "TOXIC")) + \
			list(read_corpus("non_toxic.txt", "NONTOXIC")) + \
			list(read_corpus("test.txt", "TEST"))


shuffle(sentences)

# Now we can feed the data files to LabeledLineSentence.
# 	LLS takes a dictionary with keys as the file names and and values the special prefixes
#		sentences from that document
#	Prefixes must be unique
# sources = {'toxic.txt':'TRAIN_TOXIC', 
# 		   'non_toxic.txt':'TRAIN_NONTOXIC'}
# sentences = LabeledLineSentence(sources)


model = Doc2Vec(dm = 1,
				dm_concat = 1,
				min_count = 2, 
				window = 5,
				size = 300,
				negative = 5,
				hs = 0,
				workers = cores)



model.build_vocab(sentences)


model.train(sentences, 
			total_examples = model.corpus_count,
			epochs = 20)




model.save('./toxic.d2v')

