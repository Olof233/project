import json
import os
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import nltk
from nltk.corpus import stopwords

from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.retrievers import EnsembleRetriever
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
from sklearn.feature_extraction.text import CountVectorizer

# Project imports
from retrieval.bm25 import bm25retriever, chinese_tokenizer
from retrieval.bm25S import bm25sretriever
from preprocessing.clean import remove_symbols

# ==========================================
# 🔧 核心工具：批量追加写入
# ==========================================
def append_batch_to_jsonl(data_list, filepath):
    """将一个列表的数据一次性追加写入文件"""
    if not data_list:
        return

    # 使用 'a' (append) 模式
    with open(filepath, 'a', encoding='utf-8') as f:
        for data in data_list:
            # 处理 LangChain 对象转字符串
            response_text = data['response']
            if hasattr(response_text, 'content'):
                response_text = response_text.content
            else:
                response_text = str(response_text)
            
            entry = {
                'response': response_text,
                'answer': data['answer']
            }
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')

def run_pipeline():
    # --- 配置 ---
    input_file = 'data_clean/questions/Mainland/test.jsonl'
    output_file = 'test_results.jsonl'
    
    # ⏱️ 批量设置：50条 ≈ 2~3分钟 (取决于推理速度)
    BATCH_SIZE = 50  
    
    # 性能参数
    workers = 4             
    context_window = 4096   
    max_doc_length = 800    # 截断长度

    # ==========================================
    # Phase 1: 准备工作
    # ==========================================
    print("Step 1/4: Loading resources...")
    
    # 如果文件存在，先重命名备份，防止追加到旧文件里
    if os.path.exists(output_file):
        print(f"⚠️  Backup existing {output_file} -> {output_file}.bak")
        os.rename(output_file, f"{output_file}.bak")

    try:
        stopwords.words('chinese')
    except LookupError:
        nltk.download('stopwords')
    
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    raw_data = [json.loads(line) for line in lines]
    questions = [d['question'] for d in raw_data]
    print(f"Loaded {len(questions)} items.")

    print("Loading KeyBERT (MPS)...")
    embed_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device='mps')
    kw_model = KeyBERT(embed_model)

    # ==========================================
    # Phase 2: 批量提取关键词
    # ==========================================
    print("Step 2/4: Extracting keywords...")
    clean_questions = [remove_symbols(q) for q in questions]
    vectorizer = CountVectorizer(tokenizer=chinese_tokenizer)
    
    try:
        keywords_list = kw_model.extract_keywords(clean_questions, vectorizer=vectorizer, top_n=1)
        extracted_queries = [kws[0][0] if kws else clean_questions[i] for i, kws in enumerate(keywords_list)]
    except Exception as e:
        print(f"Extraction error: {e}. Using original queries.")
        extracted_queries = clean_questions

    del kw_model, embed_model 

    # ==========================================
    # Phase 3: 检索文档 (CPU)
    # ==========================================
    print("Step 3/4: Retrieving documents...")
    bm25 = bm25retriever(k=2)
    bm25s = bm25sretriever(k=2)
    ensemble = EnsembleRetriever(retrievers=[bm25, bm25s], weights=[0.5, 0.5])
    
    llm_inputs = []
    
    def retrieve_single(idx):
        try:
            query = extracted_queries[idx]
            docs = ensemble.invoke(query)
            # 截断逻辑
            docs_text = "\n".join([d.page_content for d in docs])
            if len(docs_text) > max_doc_length:
                docs_text = docs_text[:max_doc_length] + "...(truncated)"
            
            return {
                "question": query,
                "reviews": docs_text,
                "options": str(raw_data[idx]['options']),
                "raw_data": raw_data[idx]
            }
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(tqdm(executor.map(retrieve_single, range(len(questions))), total=len(questions), desc="Retrieving"))
        llm_inputs = [r for r in results if r is not None]

    # ==========================================
    # Phase 4: 批量推理 + 批量存档
    # ==========================================
    print(f"Step 4/4: Inference ({workers}Workers)...")
    print(f"💾 Saving every {BATCH_SIZE} items (approx 2 mins).")
    
    llm = OllamaLLM(
        model="qwen3:0.6b", 
        num_thread=workers,
        num_ctx=context_window,
        keep_alive="1h"
    )
    
    template = """
你是一个擅长回答问题的专家.
这是一些相关的资料: {reviews}
这是你要回答的问题: {question}
请基于以上资料和问题，从以下选项中选择一个最合适的答案: {options}
"""
    chain = ChatPromptTemplate.from_template(template) | llm

    # 缓冲区
    results_buffer = []
    total_saved = 0

    def process_single(item):
        try:
            response = chain.invoke({
                "question": item['question'],
                "reviews": item['reviews'],
                "options": item['options']
            })
            return {
                'status': 'success',
                'response': response,
                'answer': [item['raw_data']['answer_idx'], item['raw_data']['answer'], item['raw_data']['meta_info']]
            }
        except Exception as e:
            return {'status': 'error', 'msg': str(e)}

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_single, item): i for i, item in enumerate(llm_inputs)}
        
        pbar = tqdm(as_completed(futures), total=len(futures), desc="Inferencing")
        
        for future in pbar:
            result = future.result()
            
            if result['status'] == 'success':
                # 加入缓冲区
                results_buffer.append(result)
            else:
                # 错误偶尔打印一下，不要太频繁
                pass 

            # 🔥 核心逻辑：缓冲区满了就落盘
            if len(results_buffer) >= BATCH_SIZE:
                append_batch_to_jsonl(results_buffer, output_file)
                total_saved += len(results_buffer)
                
                # 打印一个小提示（不会频繁刷屏）
                pbar.write(f"✅ Batch saved. Total: {total_saved}")
                
                # 清空缓冲区
                results_buffer = []
            
            # 更新进度条后缀
            pbar.set_postfix({"Buffer": len(results_buffer), "Saved": total_saved})

    # ==========================================
    # Phase 5: 保存剩余数据
    # ==========================================
    if results_buffer:
        append_batch_to_jsonl(results_buffer, output_file)
        total_saved += len(results_buffer)
        print(f"✅ Final batch saved.")

    print(f"\nAll Done! Total Saved: {total_saved} to {output_file}")

if __name__ == "__main__":
    run_pipeline()