# %%

# Converts the unicode file to ascii
import re

import numpy as np
import pandas as pd
import unicodedata
from sklearn.feature_extraction.text import TfidfVectorizer


def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


def preprocess_sentence(w):
    w = w.strip().lower()

    # creating a space between a word and the punctuation following it eg: "he is a boy." => "he is a boy ."
    # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping
    # -punctuation
    w = re.sub(r"([?.!,¿#@0-9])", r"", w)

    return w


samples = pd.read_csv('datasets/samples.csv')


top_tfidf = {}
NUM_TOKENS = 10
for lang in samples['LANG'].unique():
    X = samples[samples['LANG'] == lang]['TEXT'].apply(preprocess_sentence).to_numpy()

    tfidf_word = TfidfVectorizer(analyzer='word', ngram_range=(1, 1))
    X_word_transformed = tfidf_word.fit_transform(X)
    features_name = np.array(tfidf_word.get_feature_names())

    features_X = X_word_transformed.sum(axis=0)

    top_tokens = features_name[(-features_X).argsort()][0][:NUM_TOKENS]
    top_tfidf.update({
        lang: ', '.join(top_tokens)
    })

print()
pd.DataFrame.from_dict(top_tfidf, orient='index').to_csv('top_tokens.csv')
