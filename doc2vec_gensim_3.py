# Set Up and Import Statements
from gensim import utils
#from gensim.models.deprecated.doc2vec import LabeledSentence
from gensim.models import Doc2Vec

import numpy
#from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.externals import joblib

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

import os
import glob
import smart_open
from gensim.models.doc2vec import TaggedDocument


def find_original(tag_and_ind, toxic, non_toxic, test):
	splitstring = tag_and_ind.split('_')
	tag = splitstring[0]
	ind = splitstring[1]

	if (tag == "TOXIC"):
		print(toxic[int(ind)])
	elif (tag == "NONTOXIC"	):
		print(non_toxic[int(ind)])
	elif (tag == "TEST"):
		print(test[int(ind)])


WEmodel = Doc2Vec.load('./toxic.d2v')
CLmodel = joblib.load("classifier.pkl")


toxic = list(read_corpus("toxic.txt", "TOXIC"))
non_toxic = list(read_corpus("non_toxic.txt", "NONTOXIC"))
test = list(read_corpus("test.txt", "TEST"))




print("1. Test word embeddings")
print("2. Test classifier")
opt = input("Choose an option: ")


if (opt == '1'): # Word Embedding Testing 
	
	tmp = input('Enter a word to test (0 to quit): ')
	while(tmp != '0'):
		print(WEmodel.most_similar(tmp))
		tmp = input('Enter a word to test (0 to quit): ')

elif (opt == '2'): # Toxicity Classification Testing
	
	tmp = input('Enter a phrase (0 to quit): ')
	while(tmp != '0'):
		vec = WEmodel.infer_vector(utils.simple_preprocess(doc = tmp, deacc = True))
		prob = CLmodel.predict_proba(vec.reshape(1, -1))
		
		if (prob[0][0] > prob[0][1]):
			print('SVM Class: Non-toxic')
			print('Confidence: ', prob[0][0])
		else:
			print('SVM Class: Toxic')
			print('Confidence: ', prob[0][1])

		sims = WEmodel.docvecs.most_similar([vec], topn=5)
		
		for tag_and_sim in sims:
			print(tag_and_sim)
			find_original(tag_and_sim[0], toxic, non_toxic, test)

		tmp = input('Enter a phrase (0 to quit): ')

else:
	print("Not a valid option")