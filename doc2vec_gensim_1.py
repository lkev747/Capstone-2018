# Gensim Doc2Vec Tutorial - Musings of a Panda
# http://linanqiu.github.io/2015/10/07/word2vec-sentiment/

# Set Up and Import Statements
from gensim import utils
from gensim.models.deprecated.doc2vec import LabeledSentence
from gensim.models import Doc2Vec

import numpy
from random import shuffle
from sklearn.linear_model import LogisticRegression


# Input Format
#	Clean the data by converting everything to lowercase and removing punctuation
#	Each document should be on one line, separated by new lines!!!
#	Sentences should be formatted in this way:
#		[['word1', 'word2', 'word3', 'lastword'], ['label1']]
#		LabeledSentence is a tidier way to do that
#	We write our own class to handle multiple files/documents
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


# Now we can feed the data files to LabeledLineSentence.
# 	LLS takes a dictionary with keys as the file names and and values the special prefixes
#		sentences from that document
#	Prefixes must be unique
sources = {'toxic.txt':'TRAIN_TOXIC', 
		   'non_toxic.txt':'TRAIN_NONTOXIC'}
sentences = LabeledLineSentence(sources)


model = Doc2Vec(min_count = 1, 
				window = 5,
				size = 300,
				sample = 1e-4,
				negative = 5,
				workers = 8)

model.build_vocab(sentences.to_array())

model.train(sentences.sentences_perm(), 
			total_examples = model.corpus_count,
			epochs = 50)

model.save('./toxic.d2v')

