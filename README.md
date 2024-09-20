# Briefly

**Briefly** is an experimental extractive text summarizer exploiting
the principles of topic modelling.

It has: 

- a command line interface 
- a convenient web interface 
- a REST interface, and
- a client side library `talk_briefly` for the REST interface

Being an extractive summarizer, it captures the semantically important sentences
as determined by the modelling algorithm.  It is relatively 'safe' because it
does not generate sentences on its own.

**Check out the interactive demo on [huggingface](https://huggingface.co/spaces/vecspace/brieflyhf).**

# Brief user guide

## CLI

Invoking **Briefly** with the optional file argument produces an HTML-formatted
summary of the input file on the standard output which can be redirected to a file.

```
$ python3 briefly.py -h
usage: briefly.py [-h] [-f FILENAME] [-m min word count] [-t merge threshold] [-s summary size]
                  [-p no. of passes] [-i] [-v]

A program to summarize a text file.

optional arguments:
  -h, --help            show this help message and exit
  -f FILENAME, --filename FILENAME
                        Optional input file to summarize; leave out for web interface.
  -m min word count, --min_word_count min word count
                        Sentences with words having counts < this number will be dropped.[2]
  -t merge threshold, --merge_threshold merge threshold
                        Sentences closer than this threshold are merged into a single
                        subtopic.[0.01]
  -s summary size, --summary_size summary size
                        Number of sentences per summarized subtopic.[1]
  -p no. of passes, --passes no. of passes
                        Summary aggregated over these number of passes.[4]
  -i, --include_context
                        Include context before and after each summary line.[False]
  -v, --verbose         Enable verbose mode.[False]
```

## Web app

Invoking **Briefly** without the optional file argument brings up the web interface
on `http://localhost:7860`.  Use the sparkline to understand the distribution
of the extracted summary.


![Briefly web app](Doc/briefly_interface.png)

## REST API

The Briefly summarizer can be run as a service with REST endpoints.

```fastapi run briefly.py``` runs the summarizer exposing a REST API on the URL
`http://localhost:8000`.

![Briefly REST interface](Doc/rest_api.png)

- To update options to the app use the `/options` PUT endpoint.  
- To retrieve options from the app, use the `/options` GET endpoint.  
- To create a summary and retrieve it, use the `/summary` POST endpoint while
providing the text to be summarized as a JSON object.  

## The client library: `talk_briefly`

The `talk_briefly` library lets you write your own summarizer client with
just a few lines of code.

```python
from talk_briefly import BrieflyClient # Import the Briefly client module

bc = BrieflyClient("http://localhost:8000")  # Instantiate the Briefly client

print(f"Service is available: {bc.is_okay()}")  # Check if the service is available

old_opt = bc.get_options()  # Get current options and print
print(f"Current options:{old_opt}")

# Modify any current options that you choose
new_opt = old_opt.copy()
new_opt.update({"merge_threshold":0.8,"passes":1})

print(f"Options set as: {bc.set_options(new_opt)}")  # Set and print new options

with open("./Text/mayon_volcano.txt","r") as f:  # Read a file to summarize
    all_lines = f.read()
    
print(f"Summarized text:\n{bc.get_summary(all_lines)}") # Get and print the summary
```

# Installation

Inside your virtual environment, use the `requirements.txt` file
to download and install the dependencies.

> `pip -r requirements.txt`

**Optional step:** Use the script `download_model.py` to download and save
a copy of the sentence embedding BERT model locally.

# Dockerization

For creating Docker image for running both Briefly web service and REST service,
refer to instructions in the `Docker` folder.

# Notes

All arguments are set to default values to get a reasonable summary.  However,
you can experiment with the arguments within some limits.

- For small articles, setting `min_word_count` to a large value will only
capture stopwords.  You can experiment with values between 2 and 10.
Higher values of `-m` tend to give tighter summaries upto a point.  However,
documents 'too small' to be summarized will cause an error and not produce a
summary.  
- The `merge_threshold` should be as large as possible for highly cohesive
articles to get a tight summary.  
- The `summary_size` set to large values will get you non-relevant sentences
as part of your summary.  Experiment with lower values for tighter summaries.  
- Higher values of `passes` tends to give more stable/repeatable and larger
summaries across multiple invocations.  
- Use the `-i` or `--include_context` option to include context before and
after a summary line.  This helps add 'continuity' to the summary.  Note that
the context may not necessarily be part of the summary.  
- Information about top2vec is available [here](https://top2vec.readthedocs.io/en/stable/Top2Vec.html).


