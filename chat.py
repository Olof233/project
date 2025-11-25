from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from retrieval.bm25 import chinese_tokenizer
from retrieval.bm25S import tokenize
from retrieval.vector import cosineretriever
from retrieval.bm25S import bm25sretriever
from retrieval.bm25 import bm25retriever
from preprocessing.clean import remove_symbols
from preprocessing.extract import extract
from langchain_classic.retrievers import EnsembleRetriever
import json
import pickle
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

model = OllamaLLM(model="qwen3:0.6b")


template = """
你是一个擅长回答问题的专家.
这是一些相关的资料: {reviews}
这是你要回答的问题: {question}
请基于以上资料和问题，从以下选项中选择一个最合适的答案: {options}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model
answers = []
responses = []

# while True:
#     print("\n-----------")
#     question = input("Ask your question(q to quit):")
#     print("\n")
#     if question =="q":
#         break
#     retriever = bm25retriever()
#     question = extract(remove_symbols(question))
#     print(question)
#     reviews = retriever.invoke(question)
#     print('retrieval: ', reviews)
#     result = chain.invoke({"reviews": reviews,"question": question})
#     print('\nanswer: ', result)

def run(question, options, responses=responses, ifextract=True, ensemble=True):
    if ensemble:
        retriever = EnsembleRetriever(
            retrievers=[cosineretriever(), bm25sretriever(), bm25retriever()],
            weights=[0.4, 0.3, 0.3])
    else:
        retriever = bm25retriever()
    if ifextract:
        question = extract(remove_symbols(question))
    else:
        question = question
    reviews = retriever.invoke(question)
    result = chain.invoke({"reviews": reviews,"question": question, "options": options})
    responses.append(result)
    
def test(responses=responses, answers=answers):
    read_file_path = 'data_clean/questions/Mainland/test.jsonl'
    save_file_path = '.'
    with open(read_file_path, 'r+', encoding='utf-8') as f:
        print("\nGenerating answers for test set...")
        print("\n------------------")
        lines = f.readlines()
        for _, line in enumerate(tqdm(lines, desc="Reading file"), 1):
            data = json.loads(line)
            run(data['question'], str(data['options']), responses)
            answers.append([data['answer_idx'], data['answer'], data['meta_info']])
            
    result_list = []
    for j in len(answers):
        result_list.append({'response': responses[j], 'answer': answers[j]})
    with open(save_file_path + '/test_results.jsonl', 'wb+', encoding='utf-8') as wf:
        pickle.dump(result_list, wf)

    print("\nDone!")
    
        
test()