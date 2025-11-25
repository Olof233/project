def cosine(Chroma, k=5):
    retriever = Chroma.as_retriever(
                    search_kwargs={
                        "k":k
                        } 
                    
                )
    return retriever