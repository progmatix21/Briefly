Steps for text summarization:

Text summarization belongs to the field of topic modelling.

1. Take a large body of text.
2. Split the large body of text into individual sentences(or also the documents).  The text should then come as rows of text where each row is our document.
3. Convert the document into a list of sentences.
4. Feed the list of documents into a top2vec model.  The top2vec model divides our documents into a set of topics (or subtopics for our understanding).
5. Extract one or two important documents (sentences for our understanding) that lie inside each topic (for us subtopic) and output them as our summarized text.

