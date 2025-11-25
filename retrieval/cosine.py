def cosine(Chroma, k=5):
    retriever = Chroma.as_retriever(
                    search_kwargs={
                        "k":k,
                        'search_type':"similarity" # | "mmr" | "similarity_score_threshold" (need to pass score_threshold)           
                        } 
                    
                )
    return retriever