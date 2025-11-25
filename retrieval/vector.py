from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
from . import cosine
import pandas as pd
import sys 
sys.path.append("..") 

def cosineretriever(k=5):
    RETRIEVER = cosine.cosine
    file_path = "data_clean/textbooks/zh_paragraph/all_books.txt"
    embeddings = OllamaEmbeddings(model="qwen3-embedding:0.6b")

    db_location = "chroma_langchain_db"
    add_documents = not os.path.exists(db_location)

    vector_store = Chroma(
        collection_name="dataset",
        persist_directory=db_location,
        embedding_function=embeddings
    )

    if add_documents:
        row = 0
        with open(file_path, 'r+', encoding='utf-8') as f:
            documents = []
            length = 0
            for i in f.readlines():
                document = Document(page_content=i,ids=str(row))
                documents.append(document)
                length += 1
                row += 1
                if length % 1000 == 0:
                    vector_store.add_documents(documents)
                    documents = []
                    print(length)

    retriever = RETRIEVER(vector_store, k=k)
    return retriever