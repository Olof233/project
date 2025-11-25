from langchain_community.retrievers import BM25Retriever
import jieba
from typing import List
from nltk.corpus import stopwords
from langchain_core.documents import Document
import pickle
import os
import sys 
sys.path.append("..") 


def chinese_tokenizer(text: str) -> List[str]:
    tokens = jieba.lcut(text)
    return [token for token in tokens if token not in stopwords.words('chinese')]


def bm25retriever(k=5):
    
    db_location = "bm25_db"
    add_documents = not os.path.exists(db_location)
    
    if add_documents:
        file_path = "data_clean/textbooks/zh_paragraph/all_books.txt"
        with open(file_path, "r", encoding="utf-8") as f:
            Docs = ([Document(page_content=i) for i in f.readlines()])

        bm25retriever = BM25Retriever.from_documents(
            Docs,
            k = k,
            preprocess_func=chinese_tokenizer
        )
        
        with open("bm25_db\\bm25_retriever.pkl", "wb") as f:
            pickle.dump(bm25retriever, f)
    
    else:
        with open("bm25_db\\bm25_retriever.pkl", "rb") as f:
            bm25retriever = pickle.load(f)
            
        bm25retriever.preprocess_func = chinese_tokenizer
            
    return bm25retriever