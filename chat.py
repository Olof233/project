from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from retrieval.bm25 import chinese_tokenizer
from retrieval.vector import cosineretriever
from retrieval.bm25S import bm25sretriever
from retrieval.bm25 import bm25retriever

model = OllamaLLM(model="qwen3:0.6b")


template = """
你是一个擅长回答问题的专家.
这是一些相关的资料: {reviews}
这是你要回答的问题: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

while True:
    print("\n-----------")
    question = input("Ask your question(q to quit):")
    print("\n")
    if question =="q":
        break
    # retriever = cosineretriever()
    retriever = bm25retriever()
    # reviews = bm25sRetriever(question)
    reviews = retriever.invoke(question)
    print('retrieval: ', reviews)
    result = chain.invoke({"reviews": reviews,"question": question})
    print('\nanswer: ', result)