import json
import os
import time
import math
import multiprocessing
from mlx_lm import load, generate

# ================= 配置区域 =================
INPUT_FILE = 'data_clean/questions/Mainland/test.jsonl'
OUTPUT_FILE = './test_results_norag_parallel.jsonl'
# 保持和你单进程代码一致的模型 ID
MODEL_ID = "Qwen/Qwen3-0.6B-MLX-4bit"

# 进程数：M2 Max 建议设置为 8 到 10
NUM_WORKERS = 8       

# 提示词模板 (保持与你的 chat_pure_mlx.py 一致)
TEMPLATE = """
你是一个擅长回答问题的专家.
这是你要回答的问题: {question}
请基于以上问题，从以下选项中选择一个最合适的答案: {options}
"""

def worker_task(worker_id, data_chunk):
    """
    子进程工作函数：
    每个进程独立加载模型，处理分配到的数据块。
    """
    print(f"🔧 Worker {worker_id}: Loading model...")
    
    # ⚠️ 关键：模型必须在子进程内部加载，不能在主进程加载后传递
    # trust_remote_code=True 以防万一，通常 mlx 模型不需要
    model, tokenizer = load(MODEL_ID, tokenizer_config={"trust_remote_code": True})
    
    results = []
    print(f"🚀 Worker {worker_id}: Processing {len(data_chunk)} items...")
    
    start_t = time.time()
    
    for idx, item in enumerate(data_chunk):
        try:
            # 1. 构造 Prompt (保持原有逻辑)
            prompt_content = TEMPLATE.format(
                question=item['question'],
                options=item['options']
            )
            
            # 2. 应用聊天模板
            messages = [{"role": "user", "content": prompt_content}]
            prompt_text = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # 3. 生成
            # max_tokens=512 保持一致
            response_text = generate(
                model, 
                tokenizer, 
                prompt=prompt_text, 
                max_tokens=512, 
                verbose=False
            )
            
            # 4. 收集结果
            results.append({
                'response': response_text,
                'answer': [item['answer_idx'], item['answer'], item.get('meta_info', '')]
            })
            
            # 简单的进度提示
            if (idx + 1) % 50 == 0:
                print(f"✅ Worker {worker_id}: {idx + 1}/{len(data_chunk)} done")
                
        except Exception as e:
            print(f"❌ Worker {worker_id} Error at item {idx}: {e}")
            # 即使出错也保留记录，避免数据错位
            results.append({
                'response': "ERROR",
                'answer': [item['answer_idx'], item['answer'], item.get('meta_info', '')]
            })
            
    total_t = time.time() - start_t
    print(f"🏁 Worker {worker_id} finished. Speed: {len(data_chunk)/total_t:.2f} it/s")
    return results

def main():
    # ⚠️ 必须设置 spawn，否则 Metal 会报错
    multiprocessing.set_start_method('spawn', force=True)
    
    # 1. 读取数据
    print(f"📦 Loading data from {INPUT_FILE}...")
    questions = []
    if os.path.exists(INPUT_FILE):
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    questions.append(json.loads(line))
    else:
        print(f"❌ Input file not found: {INPUT_FILE}")
        return

    total_items = len(questions)
    print(f"📊 Total items: {total_items}. Launching {NUM_WORKERS} workers.")
    
    # 2. 切分数据
    chunk_size = math.ceil(total_items / NUM_WORKERS)
    chunks = [questions[i:i + chunk_size] for i in range(0, total_items, chunk_size)]
    
    # 准备参数 [(id, chunk), (id, chunk), ...]
    tasks = [(i, chunk) for i, chunk in enumerate(chunks)]
    
    start_global = time.time()
    
    # 3. 并行执行
    with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
        # starmap 自动解包参数传给 worker_task
        results_nested = pool.starmap(worker_task, tasks)
    
    # 4. 合并结果
    final_results = [item for sublist in results_nested for item in sublist]
    
    # 5. 保存
    print(f"💾 Saving {len(final_results)} results to {OUTPUT_FILE}...")
    if os.path.exists(OUTPUT_FILE):
         os.rename(OUTPUT_FILE, f"{OUTPUT_FILE}.bak")
         
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for res in final_results:
            f.write(json.dumps(res, ensure_ascii=False) + '\n')
            
    total_time = time.time() - start_global
    print(f"\n🎉 All Done! Total time: {total_time:.2f}s")
    print(f"⚡ Aggregate Speed: {total_items/total_time:.2f} it/s")

if __name__ == "__main__":
    main()