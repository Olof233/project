def cosine(Chroma):
    retriever = Chroma.as_retriever(
                    search_kwargs={"k":5} #将查找5个
                )
    return retriever