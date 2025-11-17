import bm25s
from bm25s.tokenization import Tokenized
import jieba
from typing import List, Union
from tqdm.auto import tqdm
from langchain_core.documents import Document

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

def bm25sretriever(query):
    bm25s.tokenize = tokenize
    file_path = "data_clean/textbooks/zh_paragraph/all_books.txt"
    
    if ADD:
        with open(file_path, 'r+', encoding='utf-8') as f:
            corpus = f.readlines()

        corpus_tokens = bm25s.tokenize(corpus)
        retriever = bm25s.BM25(corpus=corpus)
        retriever.index(corpus_tokens)
        retriever.save("bm25s_db")
        query_tokens = bm25s.tokenize(query)
        docs, _ = retriever.retrieve(query_tokens, k=5)
    
    else:
        retriever = bm25s.BM25.load("bm25s_db", load_corpus=True)
        query_tokens = bm25s.tokenize(query)
        docs, _ = retriever.retrieve(query_tokens, k=5)

    print(docs[0], type(docs[0]))
    results = [Document(page_content=i['text']) for i in docs[0]]
    return results