from keybert import KeyBERT
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
import sys
sys.path.append("..")


def extract(text, n_range=((1, 3))):
    model = SentenceTransformer('bert-base-chinese-sentence-transformer-xnli-zh')
    kw_model = KeyBERT(model)
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=n_range, stop_words=stopwords.words('chinese'))
    return keywords[0][0]
