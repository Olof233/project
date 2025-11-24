from keybert import KeyBERT
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
import sys
sys.path.append("..")
from retrieval.bm25 import chinese_tokenizer
from sklearn.feature_extraction.text import CountVectorizer



def extract(text):
    vectorizer = CountVectorizer(tokenizer=chinese_tokenizer, ngram_range=(1, int(len(chinese_tokenizer(text))/2)))
    model = SentenceTransformer('bert-base-chinese-sentence-transformer-xnli-zh', local_files_only=True)
    kw_model = KeyBERT(model)
    keywords = kw_model.extract_keywords(text, vectorizer=vectorizer)
    print(keywords, '\n')
    return keywords[0][0]
