#!/usr/bin/env python3

'''
tldr.py is an extractive command line text summarizer.  It uses topic
modelling at the document level rather than at the corpus level.  The
current algorithm being used is top2vec.

'''
import os
import sys
import re
from top2vec import Top2Vec
import numpy as np
import argparse
from abc import ABC, abstractmethod
# load a text file

class Formatter(ABC):
	"""
	Creates the Formatter Abstract Base Class
	"""
	@abstractmethod
	def getSummary():
		pass


class Strategy(ABC):
	"""
	Creates the Strategy Abstract Base Class
	"""
	@abstractmethod
	def executeStrategy():
		pass


class SummarizerContext():
	# Aggregates a context to do the summarization
	def __init__(self,filename):
		self._filename = filename
		self._text_prepping_strategy = None
		self._summarizing_strategy = None
		self._formatter = None
				
	def set_text_prepping_strategy(self,strategy_obj):
		self._text_prepping_strategy = strategy_obj	
	
	def set_summarizing_strategy(self,strategy_obj):
		self._summarizing_strategy = strategy_obj
		
	def set_formatter(self,formatter_obj):
		self._formatter = formatter_obj
		
	def summarize(self):
		doc_list = _text_prepping_strategy.executeStrategy(args.filename)
		summary_tuples = self._summarizing_strategy.executeStrategy(doc_list)
		return self._formatter.getSummary(summary_tuples)
		

		
class HTMLFormatter(Formatter):
	"""
	This is a subclass of the Formatter class.
	Abstract methods of the Formatter class are implemented here.
	"""
	# Summary is provided as a tuple of three lists
	# doc_ids, sentences, scores
	# We want just the sentences
	def __init__(self):
		pass
		
	def getSummary(self,summary):
		_,self.sentences,_ = list(zip(*summary))
		
		preHTML='''
		<!doctype HTML>
		<head></head>
		<title>Briefly: summary</title>
		<body>
		<h1>Briefly: summary</h1>
		<ul>
		'''
		postHTML='''
		</ul>
		<small style="font-size="5px;">Generated with <a href="https://github.com/progmatix21/Briefly">Briefly</a></small>
		</body>		
		</html>
		'''
		tagged_sents = ['<li>' + sent + '</li>' for sent in self.sentences]
		return(preHTML+" ".join(tagged_sents)+postHTML)
		

class Strategy_top2vec(Strategy):
	"""
	This is a subclass of Strategy base class.
	Abstract methods of Strategy base class are implemented here.
	"""
	def __init__(self):
		# min cluster size should be atleast 2.  We keep it fixed and control
		# the clustering with the merge threshold.
		self._hdbscan_args_dict = {'min_cluster_size':2,
		'cluster_selection_epsilon':args.merge_threshold,
		'cluster_selection_method':'eom'}
		
		
	def _init_model(self):
		# Initialize the topic modelling algorithm
		self._model = Top2Vec(self._document_list, min_count=args.min_word_count, hdbscan_args=self._hdbscan_args_dict)
	
		self._num_topics = self._model.get_num_topics()
		self.top_docs_per_topic = args.summary_size
		self.topic_sizes,self.topic_nums = self._model.get_topic_sizes()	
	
	def executeStrategy(self,document_list):
		
		self._document_list = document_list
		self._init_model()
				
		meta_documents = []
		meta_document_scores = []
		meta_document_ids = []
		
		for t in range(self._num_topics):
			# Get documents, document scores and document IDs for all documents
			documents, document_scores, document_ids = self._model.search_documents_by_topic(
				topic_num=t, num_docs=min(self.topic_sizes[t],self.top_docs_per_topic))
			# Append the document ID, document and score for all documents to their
			# respective meta arrays
			meta_documents.append(documents)
			meta_document_scores.append(document_scores)
			meta_document_ids.append(document_ids)
			
		# Flatten the meta arrays to get a continuous array
		meta_document_ids = [item for sublist in meta_document_ids for item in sublist]
		meta_document_scores = [item for sublist in meta_document_scores for item in sublist]
		meta_documents = [item for sublist in meta_documents for item in sublist]
		
		# Return the summary as a tuple sorted on document ID.
		return sorted(zip(meta_document_ids, meta_documents, meta_document_scores ))
		
		'''
		for doc_id, doc, score  in sorted(zip(meta_document_ids, meta_documents, meta_document_scores )):
			#print(f"Document: {doc_id}, Score: {score} >>>")
			print(f"{doc}.",end="  ")
			#print("----------------")
		print()
		'''

class Strategy_text_prep(Strategy):
	"""
	This is a subclass of Strategy base class.
	Abstract methods of Strategy base class are implemented here.
	"""
	'''
	def __init__(self,filename):
		# Create a text blob out of file
		self._text_blob = self._make_blob(filename)
	'''
	def _make_blob(self,filename):
		# Read text and return a text blob
		text_blob = ""
		try:
			with open(os.path.join(filename),"r") as fileinput:
				for line in fileinput.readlines():
					text = re.sub('\n',' ',line)   		
					text_blob = text_blob+text				
			return text_blob

		except:
			print("Please provide a valid file.")
			sys.exit()
			
	def _make_sentence_list(self,text_blob):
		# Given a text blob, return a list of sentences
		sentence_list = []
		split_pat = r'[.!?]\s+'
		
		# alternative approach.  Split on a regex
		sent_matches = re.split(split_pat, text_blob)
		return np.array(sent_matches)
		
	def executeStrategy(self,filename):
		text_blob = self._make_blob(filename)
		sentence_list = self._make_sentence_list(text_blob)
		return sentence_list
		#return self._make_sentence_list(self._text_blob)
	
def parseArgs():
	# Argument parser
	parser = argparse.ArgumentParser(description="A program to summarize a text file.")

	# Add a positional argument	
	parser.add_argument("filename", type=str, help="The name of the file to summarize")
	
	# Define the optional arguments

	parser.add_argument("-m","--min_word_count",metavar='',type=int, default=2, 
	help="Sentences with words having counts < this number will be dropped.")
	
	parser.add_argument("-t", "--merge_threshold", metavar='', type=float, default=0.0, 
	help="Sentences closer than this threshold are merged into a single subtopic.")

	parser.add_argument("-s", "--summary_size", metavar='', type=int, default=3, 
	help="Number of sentences per summarized subtopic.")	
	
	# return the parsed args
	return parser.parse_args()

if __name__ == "__main__":
	
	args = parseArgs()

	#documents = Strategy_text_prep(args.filename)
	
	# Testing the Strategy_text_prep class
	# Opening the file and creating the text blob is done here.
	text_prepper = Strategy_text_prep()
	
	# Get document/sentence list (the long list of sentences or docs)
	doc_list = text_prepper.executeStrategy(args.filename)
	
	
	
	#print("Document list: ",doc_list)
	# Instantiate the top2vec class
	my_top2vec = Strategy_top2vec()
	#print("Number of topics: ",my_top2vec._num_topics)
	my_summary = my_top2vec.executeStrategy(doc_list)
	
	my_formatter = HTMLFormatter()
	my_formatted_summary = my_formatter.getSummary(my_summary)
	print(my_formatted_summary)
