#!/usr/bin/env python3

'''
briefly.py is an extractive command line text summarizer.  It uses topic
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
		
	def summarize(self,n_passes):
		doc_list = self._text_prepping_strategy.executeStrategy(args.filename)
		
		summary_tuples_set = set()   # Initialize set to accumulate results
		# of multiple passes
		summary_tuples_list = []  # Final tuples list to pass to
		# formatter
		
		# get the summaries over n passes and aggregate them (by sorting) 
		for _ in range(n_passes):			
			# Each time, the model is re-initialized			
			summary_tuples = self._summarizing_strategy.executeStrategy(doc_list)
			summary_tuples_set.update(set(summary_tuples))
		
		# Create a sorted list of summary to give to formatter
		summary_tuples_list = sorted(list(summary_tuples_set))
		condensation_ratio = round(len(summary_tuples_set)/len(doc_list),2)
		return self._formatter.getSummary(summary_tuples_list,condensation_ratio)
		

		
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
		
	def getSummary(self,summary,condensation_ratio):
		_,self.sentences = list(zip(*summary))
		
		c_ratio = condensation_ratio
		
		preHTML='''
		<!doctype HTML>
		<head></head>
		<title>Briefly: summary</title>
		<body>
		<h1>Briefly: summary</h1>
		<ul>
		'''
		postHTML=f'''
		</ul>
		<small style="font-size=5px;">Condensed to {c_ratio}</small><br/>
		<small style="font-size=5px;">Generated with <a href="https://github.com/progmatix21/Briefly">Briefly</a></small>
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
		self._model = Top2Vec(self._document_list, min_count=args.min_word_count, 
		hdbscan_args=self._hdbscan_args_dict,verbose=args.verbose)
				
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
		
		# Return the summary as a list of tuples sorted on document ID.
		# Hold back document scores because it interferes with multi-pass
		# summarization
		return sorted(zip(meta_document_ids, meta_documents))
	

class Strategy_text_prep(Strategy):
	"""
	This is a subclass of Strategy base class.
	Abstract methods of Strategy base class are implemented here.
	"""

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

	m_default = 2
	parser.add_argument("-m","--min_word_count",metavar='min word count',type=int, default=m_default, 
	help=f"Sentences with words having counts < this number will be dropped.[{m_default}]")
	
	t_default = 0.01
	parser.add_argument("-t", "--merge_threshold", metavar='merge threshold', type=float, default=t_default, 
	help=f"Sentences closer than this threshold are merged into a single subtopic.[{t_default}]")

	s_default = 1
	parser.add_argument("-s", "--summary_size", metavar='summary size', type=int, default=s_default, 
	help=f"Number of sentences per summarized subtopic.[{s_default}]")	

	p_default = 4
	parser.add_argument("-p","--passes", metavar='no. of passes', type=int, default=p_default,
	help=f"Summary aggregated over these number of passes.[{p_default}]")
		
	parser.add_argument("-v","--verbose", action='store_true', default=False,
	help=f"Enable verbose mode.")

	# return the parsed args
	return parser.parse_args()

class Summarizer():
	# Creates the summarizer context and gets the summary.
	
	def __init__(self):
		# Create a summarizer context
		self.my_summarizer_context = SummarizerContext(args.filename)
		
		# Create a text prepping strategy object and set it
		text_prepper_strategy = Strategy_text_prep()
		self.my_summarizer_context.set_text_prepping_strategy(text_prepper_strategy)
		
		# Create a summarizing strategy and set it
		top2vec_strategy = Strategy_top2vec()
		self.my_summarizer_context.set_summarizing_strategy(top2vec_strategy)
		
		# Create a formatter and set it
		html_formatter = HTMLFormatter()
		self.my_summarizer_context.set_formatter(html_formatter)
	
	
	def getSummary(self):
		# Give the optional n_passes argument, default 4
		return self.my_summarizer_context.summarize(args.passes)
	

if __name__ == "__main__":
	
	args = parseArgs()	

	my_summarizer = Summarizer()
	
	print("working... ", end='', file=sys.stderr)
	summarized_text = my_summarizer.getSummary()
	print("done. ", file=sys.stderr)

	print(summarized_text)
