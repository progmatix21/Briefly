# Briefly

**Briefly** is a simple command line extractive text summarizer exploiting
the principles of topic modelling.  It produces HTML-formatted summary on the
standard output.  You can redirect this to a file.

# Usage

```
$ python3 briefly.py -h
usage: briefly.py [-h] [-m min word count] [-t merge threshold] [-s summary size] filename

A program to summarize a text file.

positional arguments:
  filename              The name of the file to summarize

optional arguments:
  -h, --help            show this help message and exit
  -m min word count, --min_word_count min word count
                        Sentences with words having counts < this number will be dropped.[2]
  -t merge threshold, --merge_threshold merge threshold
                        Sentences closer than this threshold are merged into a single
                        subtopic.[0.01]
  -s summary size, --summary_size summary size
                        Number of sentences per summarized subtopic.[2]


```

All arguments are set to default values to get a reasonable summary.  However,
you can experiment with the arguments within some limits.

- For small articles, setting `-m` to a large value will only capture stopwords.
You can experiment with values between 2 and 10.
- The `merge_threshold` should be as small as possible for highly cohesive
articles.
- The `summary_size` set to large values will get you non-relevant sentences
as part of your summary.

# Installation requirements

- [top2vec](https://top2vec.readthedocs.io/en/stable/Top2Vec.html#installation)

