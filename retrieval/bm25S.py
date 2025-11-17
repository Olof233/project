import bm25s
from bm25s.tokenization import Tokenized
import jieba
from typing import List, Union
from tqdm.auto import tqdm
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

import os
import sys 
sys.path.append("..") 

ADD = not os.path.exists("bm25s_db")

def tokenize(
    texts,
    return_ids: bool = True,
    show_progress: bool = True,
    leave: bool = False,
) -> Union[List[List[str]], Tokenized]:
    if isinstance(texts, str):
        texts = [texts]

    corpus_ids = []
    token_to_index = {}

    for text in tqdm(
        texts, desc="Split strings", leave=leave, disable=not show_progress
    ):

        splitted = jieba.lcut(text)
        doc_ids = []

        for token in splitted:
            if token not in token_to_index:
                token_to_index[token] = len(token_to_index)

            token_id = token_to_index[token]
            doc_ids.append(token_id)

        corpus_ids.append(doc_ids)

    unique_tokens = list(token_to_index.keys())

    vocab_dict = token_to_index

    if return_ids:
        return Tokenized(ids=corpus_ids, vocab=vocab_dict)

    else:
        reverse_dict = unique_tokens
        for i, token_ids in enumerate(
            tqdm(
                corpus_ids,
                desc="Reconstructing token strings",
                leave=leave,
                disable=not show_progress,
            )
        ):
            corpus_ids[i] = [reverse_dict[token_id] for token_id in token_ids]

        return corpus_ids

# def bm25sRetriever(query):
#     bm25s.tokenize = tokenize
#     file_path = "data_clean/textbooks/zh_paragraph/all_books.txt"
    
#     if ADD:
#         with open(file_path, 'r+', encoding='utf-8') as f:
#             corpus = f.readlines()

#         corpus_tokens = bm25s.tokenize(corpus)
#         retriever = bm25s.BM25(corpus=corpus)
#         retriever.index(corpus_tokens)
#         retriever.save("bm25s_db")
#         query_tokens = bm25s.tokenize(query)
#         docs, _ = retriever.retrieve(query_tokens, k=5)
    
#     else:
#         retriever = bm25s.BM25.load("bm25s_db", load_corpus=True)
#         query_tokens = bm25s.tokenize(query)
#         docs, _ = retriever.retrieve(query_tokens, k=5)

#     print(docs[0], type(docs[0]))
#     results = [Document(page_content=i['text']) for i in docs[0]]
#     return results

class BM25SRetriever(BaseRetriever):
    """Custom BM25S Retriever that integrates with LangChain"""
    
    data_path: str = Field(default="data_clean/textbooks/zh_paragraph/all_books.txt")
    db_path: str = Field(default="bm25s_db")
    k: int = Field(default=5)
    
    _retriever = None
    _is_initialized = False
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._initialize_retriever()
    
    def _initialize_retriever(self):
        """Initialize the BM25S retriever, building index if needed"""
        if self._is_initialized:
            return
            
        bm25s.tokenize = tokenize
        add_documents = not os.path.exists(self.db_path)
        
        if add_documents:
            print("Building BM25S index...")
            with open(self.data_path, 'r', encoding='utf-8') as f:
                corpus = f.readlines()
            
            corpus_tokens = bm25s.tokenize(corpus)
            self._retriever = bm25s.BM25(corpus=corpus)
            self._retriever.index(corpus_tokens)
            self._retriever.save(self.db_path)
        else:
            print("Loading existing BM25S index...")
            self._retriever = bm25s.BM25.load(self.db_path, load_corpus=True)
        
        self._is_initialized = True
    
    def _get_relevant_documents(self, query: str, *, run_manager) -> List[Document]:
        """Retrieve documents relevant to the query"""
        if not self._is_initialized or self._retriever is None:
            self._initialize_retriever()
        
        bm25s.tokenize = tokenize
        query_tokens = bm25s.tokenize(query)
        docs, _ = self._retriever.retrieve(query_tokens, k=self.k)
        
        results = []
        for doc in docs:
            if isinstance(doc, dict) and 'text' in doc:
                results.append(Document(page_content=doc['text']))
            elif isinstance(doc, str):
                results.append(Document(page_content=doc))
            else:
                results.append(Document(page_content=str(doc)))
        
        return results
    
    async def _aget_relevant_documents(self, query: str, *, run_manager) -> List[Document]:
        """Async version of retrieve - can be the same as sync for now"""
        return self._get_relevant_documents(query, run_manager=run_manager)

def bm25sretriever(
    data_path: str = "data_clean/textbooks/zh_paragraph/all_books.txt", 
    db_path: str = "bm25s_db", 
    k: int = 5
) -> BM25SRetriever:
    return BM25SRetriever(data_path=data_path, db_path=db_path, k=k)