#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# Briefly:briefly.py - Extractive summarizer with command line, web, REST interface.
# Copyright (C) 2024  Atrij Talgery: github.com/progmatix21
# SPDX-License-Identifier: AGPL-3.0-or-later
# https://www.gnu.org/licenses/agpl.txt
# https://spdx.org/licenses/AGPL-3.0-or-later.html
"""

"""
briefly.py is an extractive command line text summarizer.  It uses topic
modelling at the document level rather than at the corpus level.  The
current algorithm being used is top2vec.
"""

import sys

import argparse
from collections import namedtuple

import matplotlib
matplotlib.use("Agg")

import brieflycore

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

# Function to convert this summarizer to gradio app
def summarizer_app(min_word_count,merge_threshold,passes,summary_size,include_context,input_text):
    
    # store the arguments from widget/sliders
    w_args = Widget_args(input_text,merge_threshold,min_word_count,passes,summary_size,include_context,False)
    # verbose flag is hard coded to False above
    # send the input text instead of filename
    #"filename merge_threshold min_word_count passes summary_size include_context verbose"
    
    web_summarizer = brieflycore.Summarizer(w_args,web=True)
    #Print web status message here
    summarized_text = web_summarizer.getSummary()
    
    return summarized_text

# Special code for using Briefly in the module mode for FAST API
# Note that the following two blocks are mutually exclusive
if __name__ != "__main__":
    
    from fastapi import FastAPI
    from pydantic import BaseModel

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
        rest_summarizer = brieflycore.Summarizer(options_resource,web=True,formatter='plain')
        summarized_text = rest_summarizer.getSummary()        

        return {"text":summarized_text}

    
    # Client receives current options in the system
    @app.get("/options")
    async def rest_get_options() -> Options:
        '''Get current option values'''
        # Blankout the file text from options. File text is set in rest_get_summary()
        options_copy = options_resource.model_copy(update={'filename':""})
        return options_copy

    
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
        
    else:    # Stay with the command line interface
        my_summarizer = brieflycore.Summarizer(args,web=False,formatter='html')
        
        print("working... ", end='', file=sys.stderr, flush=True)
        summarized_text = my_summarizer.getSummary()
        print("done. ", file=sys.stderr)

        print(summarized_text)
