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
from collections import namedtuple
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import base64

from io import BytesIO

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
		doc_list = self._text_prepping_strategy.executeStrategy(Summarizer.args.filename)
		
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
	
	# Adapted from: https://www.markhneedham.com/blog/2017/09/23/python-3-create-sparklines-using-matplotlib/
	def sparkline(self, summary_lines, n_bins, vline, figsize=(8,0.5)):
		"""
		Returns an HTML image tag containing a base 64 encoded sparkline
		style plot of the summary.
		"""
		
		fig, ax = plt.subplots(1,1, figsize=figsize)
		
		ax.hist(summary_lines, bins=n_bins)  #Plot a histogram with 100 bins
		
		for k,v in ax.spines.items():
			if k == 'bottom':
				v.set_visible(True)
			else:
				v.set_visible(False)
		
		# Line to mark the end of summary
		plt.axvline(vline, color='0.8', linestyle='--')  
		
		# Label the y axis with values
		ax.set_yticks([])
		
		# Other styling of the image
		img = BytesIO()
		plt.savefig(img, transparent=True, bbox_inches='tight')
		img.seek(0)
		plt.close()
		
		# Return format of image
		return base64.b64encode(img.read()).decode("UTF-8")
		
	def getSummary(self,summary,condensation_ratio):
		# Input: summary tuples: (doc_id, sentence)
		# returns: HTML formatted summary
		sent_ids,self.sentences = list(zip(*summary))
		
		c_ratio = condensation_ratio
		
		preHTML='''
		<!doctype HTML>
		<head></head>
		<title>Briefly: summary</title>
		<body>
		<h1>Briefly: summary</h1>
		<ul>
		'''
		try:
			original_doc_length = math.ceil(len(sent_ids)/c_ratio)
		except:
			original_doc_length = np.nan
			
		n_bins = min(100,original_doc_length)
		
		# sparkline implementation
		
		postHTML=f'''
		<div align="center"><img src="data:image/png;base64,{self.sparkline(sent_ids,n_bins,original_doc_length)}"><div>
		</ul>
		<small style="font-size=5px;">Condensed to {c_ratio}</small><br/>
		<small style="font-size=5px;">Generated with <a href="https://github.com/progmatix21/Briefly">Briefly</a></small>
		</body>		
		</html>
		'''
		tagged_sents = ['<li>' + sent + "." + '</li>' for sent in self.sentences]
		return(preHTML+" ".join(tagged_sents)+postHTML)

		
class PlainFormatter(Formatter):
	"""
	This is the subclass of the Formatter class for plain formatting.
	Abstract methods of the Formatter class are implemented here.
	Use with REST API.
	"""
	def __init__(self):
		pass
		
	def getSummary(self,summary,condensation_ratio):
		_,self.sentences = list(zip(*summary))
		return(".  ".join(self.sentences)+".")
	
class Strategy_top2vec(Strategy):
	"""
	This is a subclass of Strategy base class.
	Abstract methods of Strategy base class are implemented here.
	"""
	def __init__(self):
		# min cluster size should be atleast 2.  We keep it fixed and control
		# the clustering with the merge threshold.
		self._hdbscan_args_dict = {'min_cluster_size':2,
		'cluster_selection_epsilon':Summarizer.args.merge_threshold,
		'cluster_selection_method':'eom'}
		self._model = None
		
		
	def _init_model(self):
		# Initialize the topic modelling algorithm
		model_location = "./Models/"
		#model_name = "all-MiniLM-L6-v2"
		model_name = "paraphrase-multilingual-MiniLM-L12-v2"
		
		if os.path.exists(model_location+model_name): # If pretrained model available
			embed_model_name = model_name 
			embed_model_path = model_location+model_name
			
		else:
			embed_model_name = "doc2vec"
			embed_model_path = None
						
		if self._model != None:  # Model already exists
			self._model.compute_topics(hdbscan_args=self._hdbscan_args_dict)
			
		else:		
			self._model = Top2Vec(self._document_list, min_count=Summarizer.args.min_word_count, 
			hdbscan_args=self._hdbscan_args_dict,verbose=Summarizer.args.verbose,
			embedding_batch_size = 4, workers=4,
			embedding_model=embed_model_name, embedding_model_path=embed_model_path)
				
		self._num_topics = self._model.get_num_topics()
		self.top_docs_per_topic = Summarizer.args.summary_size # From arguments
		self.topic_sizes,self.topic_nums = self._model.get_topic_sizes()	
	
	def executeStrategy(self,document_list):
		
		self._document_list = document_list
		self._len_document_list = len(self._document_list)
		
		error_msg = '''ERROR: Document too short.  Try with a longer document
		and/or a smaller value of merge threshold. See console for details.'''
		
		try:
			self._init_model() # Build and initialize top2vec model
		except Exception as e:
			print(f"Exception while building summary:\n{e}")
			return [(0,error_msg)]
			
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
		
		# Implementation for include context option
		# Include one ID < and one > the summarized IDs in the meta_document_ids list.
		# also include corresponding text in meta_documents list
		
		if Summarizer.args.include_context == True:
			meta_document_ids_set = set(meta_document_ids) # Create set of meta_document_ids
			# Add context IDs to the set
			for doc_id in meta_document_ids:
				if (doc_id-1) >= 0: # Avoid edge condition
					meta_document_ids_set.add(doc_id-1)
				if (doc_id+1) < self._len_document_list:
					meta_document_ids_set.add(doc_id+1)
			# Now the above set includes context IDs but they are unordered
			# We sort them into a list, this list has context IDs also
			meta_document_ids = sorted(list(meta_document_ids_set))
			# Now, build meta_documents list from scratch to include
			# context documents as well
			meta_documents = []
			for doc_id in meta_document_ids:
				meta_documents.append(self._document_list[doc_id])
			
		
		# Return the summary as a list of tuples sorted on document ID.
		# Hold back document scores because it interferes with multi-pass
		# summarization
		return sorted(zip(meta_document_ids, meta_documents))
	

class Strategy_text_prep(Strategy):
	"""
	This is a subclass of Strategy base class.
	Abstract methods of Strategy base class are implemented here.
	"""

	def _make_blob_from_text(self, mass_of_text):
		# Convert mass of text to text blob
		text_blob = re.sub('\n',' ',mass_of_text)
		return text_blob

	def _make_blob(self,filename):
		# Read text file and return a text blob
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
		
	def executeStrategy(self,file_or_text):
		
		# Filename can also be a mass of text
		if Summarizer.web == True:  # A mass of text
			text_blob = self._make_blob_from_text(file_or_text)
		else: # It is a filename
			text_blob = self._make_blob(file_or_text)
		
		sentence_list = self._make_sentence_list(text_blob)
		return sentence_list
		#return self._make_sentence_list(self._text_blob)

# A function to return default values of arguments
# Default values are hard coded inside this function

def get_default_args():
	
	Default_args = namedtuple("Arguments",
					"filename merge_threshold min_word_count passes summary_size include_context verbose")

	default_args = Default_args(None, 0.5, 2, 4, 1, False, False)
	
	return default_args

	
def parseArgs():
	# Argument parser
	parser = argparse.ArgumentParser(description="A program to summarize a text file.")

	# Get default args
	defaults = get_default_args()
	
	# Add option to switch to webapp
	parser.add_argument("-f","--filename", type=str, default=defaults.filename,
	help="Optional input file to summarize; leave out for web interface.")
	
	# Define the optional arguments

	parser.add_argument("-m","--min_word_count",metavar='min word count',type=int, default=defaults.min_word_count, 
	help=f"Sentences with words having counts < this number will be dropped.[{defaults.min_word_count}]")
	
	parser.add_argument("-t", "--merge_threshold", metavar='merge threshold', type=float, default=defaults.merge_threshold, 
	help=f"Sentences closer than this threshold are merged into a single subtopic.[{defaults.merge_threshold}]")

	parser.add_argument("-s", "--summary_size", metavar='summary size', type=int, default=defaults.summary_size, 
	help=f"Number of sentences per summarized subtopic.[{defaults.summary_size}]")	

	parser.add_argument("-p","--passes", metavar='no. of passes', type=int, default=defaults.passes,
	help=f"Summary aggregated over these number of passes.[{defaults.passes}]")
	
	parser.add_argument("-i","--include_context",action='store_true',default=defaults.include_context,
	help=f"Include context before and after each summary line.[{defaults.include_context}]")	
	
	parser.add_argument("-v","--verbose", action='store_true', default=defaults.verbose,
	help=f"Enable verbose mode.[{defaults.verbose}]")

	# return the parsed args
	return parser.parse_args()

class Summarizer():
	# Creates the summarizer context and gets the summary.
	# The argument is the object with command line arguments

	# Initialize a class variable to store command line arguments
	args = None
	web = None
	
	def __init__(self, args, web=False, formatter='html'):
		
		# Initialise the class variable with command line args
		Summarizer.args = args
		Summarizer.web = web
		
		# Create a summarizer context
		self.my_summarizer_context = SummarizerContext(Summarizer.args.filename)
		
		# Create a text prepping strategy object and set it
		text_prepper_strategy = Strategy_text_prep()
		self.my_summarizer_context.set_text_prepping_strategy(text_prepper_strategy)
		
		# Create a summarizing strategy and set it
		top2vec_strategy = Strategy_top2vec()
		self.my_summarizer_context.set_summarizing_strategy(top2vec_strategy)
		
		# Create a formatter and set it
		if formatter == 'html':
			self.my_summarizer_context.set_formatter(HTMLFormatter())
		else:
			self.my_summarizer_context.set_formatter(PlainFormatter())
		
		
	def getSummary(self):
		# Give the optional n_passes argument, default 4
		return self.my_summarizer_context.summarize(Summarizer.args.passes)
	
# Function to convert this summarizer to gradio app
def summarizer_app(min_word_count,merge_threshold,passes,summary_size,include_context,input_text):
	
	# store the arguments from widget/sliders
	w_args = Widget_args(input_text,merge_threshold,min_word_count,passes,summary_size,include_context,False)
	# verbose flag is hard coded to False above
	# send the input text instead of filename
	#"filename merge_threshold min_word_count passes summary_size include_context verbose"
	
	web_summarizer = Summarizer(w_args,web=True)
	#Print web status message here
	summarized_text = web_summarizer.getSummary()
	
	return summarized_text

# Special code for using Briefly in the module mode for FAST API
# Note that the following two blocks are mutually exclusive
if __name__ != "__main__":
	
	from fastapi import FastAPI, HTTPException
	from pydantic import BaseModel
	
	import json

	api_args = get_default_args()
	
	print("In REST API mode.")

	app = FastAPI(
	title="Briefly: An extractive summarizer",
    description="REST API interface for a summarizer service.",
    version="0.1.0",
    )
	
	# Create the options resource class
	class Options(BaseModel):
		filename: str = "" 
		merge_threshold: float = api_args.merge_threshold
		min_word_count: int = api_args.min_word_count
		passes: int = api_args.passes
		summary_size: int = api_args.summary_size
		include_context: bool = api_args.include_context
		verbose: bool = False
		
	# Create an options resource
	options_resource = Options()
		
	
	@app.get("/")
	async def greeting() -> dict[str,str]:
		message = '''
Welcome to Briefly REST API.
Endpoints are:
POST /summary
GET  /options
PUT  /options
		'''
		return {"message":message}
	
	
	# Define a class for text
	class Text(BaseModel):
		text: str = None
	
	# Create summary from client-supplied text and return summary to client
	@app.post("/summary")
	async def rest_get_summary(text_to_summarize: Text) -> Text:
		'''Create a summary from supplied text'''
		
		options_resource.filename = text_to_summarize.text
		options_resource.verbose = False  # Force verbose to false

		# Create the summarizer		
		rest_summarizer = Summarizer(options_resource,web=True,formatter='plain')
		summarized_text = rest_summarizer.getSummary()		

		return {"text":summarized_text}

	
	# Client receives current options in the system
	@app.get("/options")
	async def rest_get_options() -> Options:
		'''Get current option values'''
		
		#return {"options":options_resource}
		return options_resource

	
	# Update user's options	
	@app.put("/options")
	async def rest_put_options(options_update: Options) -> Options:
		'''Update/modify options.  Return updated options.'''
		
		# Update golden copy of options
		options_resource.__dict__.update(options_update.__dict__)
		
		return options_resource


# For webapp and command line
if __name__ == "__main__":
	
	# Parse command line arguments
	args = parseArgs()
	
	# Create a named tuple subclass to hold widget arguments
	
	Widget_args = namedtuple("Arguments",
					"filename merge_threshold min_word_count passes summary_size include_context verbose")
	

	if args.filename == None:   # no input file provided, invoke web interface
		import gradio as gr
		print("In webapp mode")
		
		demo = gr.Interface(
			fn=summarizer_app,
			
			inputs=[gr.Slider(1,10,value=args.min_word_count,step=1,label="min word count",
			info="Sentences with words having counts<this number will be dropped"),
			
			gr.Slider(0.01,0.99,value=args.merge_threshold,step=0.01,label="merge threshold",
			info="Sentences closer than this threshold are merged into a single subtopic"),
			
			gr.Slider(1,10,value=args.passes,step=1,label="no of passes",
			info="Summary aggregated over these number of passes"),
			
			gr.Slider(1,10,value=args.summary_size,step=1,label="summary size",
			info="Number of sentences per summarized subtopic"),
			
			gr.Checkbox(value=args.include_context,label="Include context",info="Include context before and after each summary line."),
			
			gr.Text(label="input text",info="Paste text to be summarized here.")],
			
			title="Briefly: an extractive summarizer",
			
			outputs=["html"]
		)
		demo.launch()
		
	else:	# Stay with the command line interface
		my_summarizer = Summarizer(args,web=False,formatter='html')
		
		print("working... ", end='', file=sys.stderr, flush=True)
		summarized_text = my_summarizer.getSummary()
		print("done. ", file=sys.stderr)

		print(summarized_text)
