import json
import os

# 配置
INPUT_FILE = 'data_clean/questions/Mainland/test.jsonl'
OUTPUT_FILE = 'batch_inference_input.jsonl'
MODEL_NAME = "doubao-pro-32k-241215" # 示例模型名称，用户上传时可能需要指定，或者此处仅为body中的占位，实际由endpoint决定？
# 根据文档，body中参数与API一致。chat API通常不需要在body里传model，而是endpoint决定。
# 但有些API实现可能兼容。为了安全，只需包含 messages, max_tokens 等参数。

TEMPLATE = """
你是一个擅长回答问题的专家.
这是你要回答的问题: {question}
请基于以上问题，从以下选项中选择一个最合适的答案: {options}
请只回答选项字母（如A、B、C、D、E），不要包含任何其他内容。
"""

def generate_batch_file():
    print(f"Loading data from {INPUT_FILE}...")
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file not found: {INPUT_FILE}")
        return

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        questions = [json.loads(line) for line in f if line.strip()]

    print(f"Processing {len(questions)} items...")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        for idx, item in enumerate(questions):
            # 构造Prompt
            prompt_content = TEMPLATE.format(question=item['question'], options=item['options'])
            
            # 构造Volcengine Batch Inference Request
            # custom_id 必须唯一
            custom_id = f"req-{idx}-{item.get('answer_idx', 'unknown')}" # 把答案idx放入id方便后续核对（如果有）
            
            request_body = {
                "messages": [
                    {"role": "user", "content": prompt_content}
                ],
                "max_tokens": 1024, # 足够输出选项了
                "temperature": 0.01, # 降低随机性
                "top_p": 1
            }
            
            batch_item = {
                "custom_id": custom_id,
                "body": request_body
            }
            
            f_out.write(json.dumps(batch_item, ensure_ascii=False) + '\n')

    print(f"Successfully generated {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_batch_file()
