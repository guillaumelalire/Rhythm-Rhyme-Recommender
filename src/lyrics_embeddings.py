import pandas as pd
import re
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')

def generate_embeddings(lyrics, song_names):
    # Remove song parts identifiers: verse, chorus, etc.
    lyrics = [re.sub(r'\[.*\]','', str(x)) for x in lyrics]
    
    nb_songs = len(lyrics)

    # Create a list of TaggedDocument objects: tokenized and lowercased lyrics, tagged with the corresponding song name
    tagged_data = [TaggedDocument(words=word_tokenize(lyrics[i].lower()), tags=[song_names[i]]) for i in range(nb_songs)]

    # Initialize a Doc2Vec model, build the vocabulary from the tagged documents and train the model
    model = Doc2Vec(vector_size=50, min_count=5, epochs=50)
    model.build_vocab(tagged_data)
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)

    # Generate embeddings for all lyrics using the trained Doc2Vec model
    document_vectors = [model.infer_vector(word_tokenize(doc.lower())) for doc in lyrics]

    return document_vectors