import json
import os
import time
import math
import multiprocessing
import re
from mlx_lm import load, generate

# ================= 配置区域 =================
INPUT_FILE = 'data_clean/questions/Mainland/test.jsonl'
OUTPUT_FILE = './test_results_norag_parallel.jsonl'
MODEL_ID = "Qwen/Qwen3-0.6B-MLX-4bit"

# M2 Max 32G 推荐配置
NUM_WORKERS = 16  
BATCH_SIZE = 24   

# 系统环境优化
os.environ["MTL_COMPUTE_PREF"] = "high-performance"  
os.environ["MLX_GPU_MEMORY_LIMIT"] = "28GB"          
os.environ["TOKENIZERS_PARALLELISM"] = "false"       

# 🔥 修改1：Prompt 优化 - 使用 One-Shot (单例) 引导，而不是生硬的 /no_think
# 0.6B 小模型需要“看例子”才能懂，而不是“听命令”
TEMPLATE = """你是一个医学专家。请阅读题目和选项，直接选出最正确的一项。

【示例】
题目：感冒的常见症状不包括？
选项：A. 鼻塞 B. 咳嗽 C. 骨折 D. 发热 E. 乏力
答案：C

【正式题目】
题目：{question}
选项：{options}
答案："""

def log(message, worker_id=None):
    """线程安全的日志输出"""
    prefix = f"[Worker {worker_id}]" if worker_id is not None else "[Main]"
    print(f"{time.strftime('%H:%M:%S')} {prefix} {message}")

def calculate_accuracy(file_path):
    """内置准确率计算函数"""
    print("\n" + "="*50)
    log(f"Starting Accuracy Calculation for {file_path}...")
    
    total = 0
    correct = 0
    valid_format = 0
    
    if not os.path.exists(file_path):
        log(f"Error: Result file {file_path} not found.")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try:
                item = json.loads(line)
                total += 1
                
                # 1. 获取模型响应
                response = item.get('response', '').strip()
                
                # 2. 获取真实标签 (兼容多种存储格式)
                # 🔥 修复：增强的 Ground Truth 提取逻辑
                raw_answer = item.get('answer', [])
                ground_truth = "N/A"
                
                if isinstance(raw_answer, list) and len(raw_answer) > 0:
                    # 优先取 answer_idx (A/B/C/D)
                    ground_truth = str(raw_answer[0]).strip()
                else:
                    # 如果只有文本，尝试从 meta_info 或其他地方找，或者暂时只统计格式
                    ground_truth = str(raw_answer).strip()

                # 3. 正则提取模型输出的选项
                match = re.search(r'([A-E])', response.split('\n')[0]) 
                
                if match:
                    pred = match.group(1)
                    valid_format += 1
                    # 只有当 ground_truth 也是 A-E 单字母时，比对才有意义
                    if pred == ground_truth:
                        correct += 1
                
            except Exception as e:
                # print(f"Error parsing line: {e}") # 减少刷屏
                pass

    # 3. 输出统计报告
    if total > 0:
        acc = (correct / total) * 100
        print("-" * 30)
        print(f"📊 Evaluation Report:")
        print(f"   Total Samples:   {total}")
        print(f"   Valid Responses: {valid_format} (Format Compliance: {valid_format/total*100:.1f}%)")
        print(f"   Correct:         {correct}")
        print(f"   Wrong:           {total - correct}")
        print(f"   ✅ Accuracy:      {acc:.2f}%")
        print("-" * 30)
    else:
        print("⚠️  No data found in result file.")

def worker_task(worker_id, data_chunk):
    """子进程：执行推理任务"""
    log("Loading model...", worker_id)
    model, tokenizer = load(MODEL_ID, tokenizer_config={"trust_remote_code": True})
    
    # 预热
    dummy_msg = [{"role": "user", "content": "A"}]
    # 尝试关闭 thinking (如果支持)，不支持则忽略
    try:
        dummy_prompt = tokenizer.apply_chat_template(dummy_msg, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        dummy_prompt = tokenizer.apply_chat_template(dummy_msg, tokenize=False, add_generation_prompt=True)
    
    generate(model, tokenizer, prompt=dummy_prompt, max_tokens=2, verbose=False)
    log("Model ready ✓", worker_id)
    
    results = []
    processed = 0
    total_items = len(data_chunk)
    
    for i in range(0, total_items, BATCH_SIZE):
        batch = data_chunk[i:i + BATCH_SIZE]
        
        for item in batch:
            try:
                # 构造 Prompt
                prompt_content = TEMPLATE.format(question=item['question'], options=item['options'])
                messages = [{"role": "user", "content": prompt_content}]
                
                # Tokenizer 处理
                try:
                    prompt_text = tokenizer.apply_chat_template(
                        messages, 
                        tokenize=False, 
                        add_generation_prompt=True,
                        enable_thinking=False 
                    )
                except TypeError:
                    prompt_text = tokenizer.apply_chat_template(
                        messages, 
                        tokenize=False, 
                        add_generation_prompt=True
                    )
                

                response = generate(
                    model, 
                    tokenizer, 
                    prompt=prompt_text, 
                    max_tokens=64,
                    verbose=False,
                )
                
                results.append({
                    'id': item.get('id'),
                    'response': response.strip(),
                    # 🔥 修改3：确保保存结构为 [answer_idx, answer_text]
                    # 优先取 item['answer_idx']，如果不存在则取 item['answer'] 的第一个元素(如果是列表)
                    'answer': [item.get('answer_idx', item.get('answer')), item.get('answer')], 
                    'meta_info': item.get('meta_info')
                })
                
            except Exception as e:
                log(f"Error: {e}", worker_id)
                results.append({'id': item.get('id'), 'response': "ERROR", 'answer': []})

        processed += len(batch)
        if worker_id == 1:
            print(f"\rProgress: [{processed}/{total_items}]", end="")
            
    return results

def main():
    multiprocessing.set_start_method('spawn', force=True)
    
    log(f"Loading data from {INPUT_FILE}...")
    if not os.path.exists(INPUT_FILE):
        log(f"Error: {INPUT_FILE} not found.")
        return
        
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        questions = [json.loads(line) for line in f if line.strip()]
    
    total_items = len(questions)
    log(f"Total items: {total_items}. Launching {NUM_WORKERS} workers...")
    
    chunk_size = math.ceil(total_items / NUM_WORKERS)
    chunks = [questions[i:i + chunk_size] for i in range(0, total_items, chunk_size)]
    tasks = [(i+1, chunk) for i, chunk in enumerate(chunks)]
    
    start_time = time.time()
    
    with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
        results_nested = pool.starmap(worker_task, tasks)
    
    final_results = [item for sublist in results_nested for item in sublist]
    
    log(f"Saving {len(final_results)} results to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for res in final_results:
            f.write(json.dumps(res, ensure_ascii=False) + '\n')
            
    total_time = time.time() - start_time
    
    print("\n" + "="*50)
    print(f"⏱️  Inference Completed in {total_time:.2f}s")
    print(f"⚡ Throughput: {total_items / total_time:.2f} items/sec")
    
    calculate_accuracy(OUTPUT_FILE)

if __name__ == "__main__":
    main()