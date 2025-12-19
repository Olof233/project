import json
import re

def process_jsonl_file(file_path):
    """
    处理 JSONL 文件，计算模型预测的准确率
    
    Args:
        file_path: JSONL 文件路径
    
    Returns:
        total_samples: 总样本数
        correct_predictions: 正确预测数
        accuracy: 准确率
    """
    total_samples = 0
    correct_predictions = 0
    results = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                
                # 获取 custom_id 字段
                custom_id = data.get('custom_id', '')
                if not custom_id:
                    print(f"警告: 第 {line_num} 行没有 custom_id 字段")
                    continue
                
                # 从 custom_id 中提取真实答案（第二个 '-' 后的字母）
                # 格式如: req-3186-E
                parts = custom_id.split('-')
                if len(parts) < 3:
                    print(f"警告: 第 {line_num} 行 custom_id 格式不正确: {custom_id}")
                    continue
                
                true_label = parts[2]  # 取第三个部分作为真实答案
                
                # 检查真实标签是否为单个字母
                if not re.match(r'^[A-E]$', true_label):
                    print(f"警告: 第 {line_num} 行提取的真实答案不是有效的字母: {true_label}")
                    continue
                
                # 获取 content 字段（模型预测结果）
                response_body = data.get('response', {}).get('body', {})
                choices = response_body.get('choices', [])
                
                if not choices:
                    print(f"警告: 第 {line_num} 行没有 choices 字段")
                    continue
                
                content = choices[0].get('message', {}).get('content', '').strip()
                
                # 提取预测标签（只取第一个字符，应为 A-E 的字母）
                if not content:
                    print(f"警告: 第 {line_num} 行 content 为空")
                    continue
                
                predicted_label = content[0].upper()
                
                # 检查预测标签是否为有效的字母
                if not re.match(r'^[A-E]$', predicted_label):
                    print(f"警告: 第 {line_num} 行预测的答案不是有效的字母: {content}")
                    continue
                
                # 记录结果
                is_correct = true_label == predicted_label
                if is_correct:
                    correct_predictions += 1
                
                total_samples += 1
                results.append({
                    'line': line_num,
                    'custom_id': custom_id,
                    'true_label': true_label,
                    'predicted_label': predicted_label,
                    'is_correct': is_correct
                })
                
            except json.JSONDecodeError:
                print(f"错误: 第 {line_num} 行 JSON 格式错误")
                continue
            except Exception as e:
                print(f"错误: 处理第 {line_num} 行时发生异常: {e}")
                continue
    
    # 计算准确率
    accuracy = correct_predictions / total_samples if total_samples > 0 else 0
    
    return total_samples, correct_predictions, accuracy, results

def print_detailed_results(results, max_display=10):
    """
    打印详细的预测结果
    
    Args:
        results: 预测结果列表
        max_display: 最多显示的错误样本数
    """
    print("\n=== 详细预测结果 ===")
    print(f"{'行号':<6} {'样本ID':<15} {'真实答案':<8} {'预测答案':<8} {'结果':<6}")
    print("-" * 50)
    
    for r in results:
        status = "✓" if r['is_correct'] else "✗"
        print(f"{r['line']:<6} {r['custom_id']:<15} {r['true_label']:<8} {r['predicted_label']:<8} {status:<6}")
    
    # 列出预测错误的样本
    errors = [r for r in results if not r['is_correct']]
    if errors:
        print(f"\n=== 预测错误的样本 ({len(errors)} 个) ===")
        for i, e in enumerate(errors[:max_display], 1):
            print(f"{i}. 第 {e['line']} 行: {e['custom_id']} - 真实: {e['true_label']}, 预测: {e['predicted_label']}")
        
        if len(errors) > max_display:
            print(f"... 还有 {len(errors) - max_display} 个错误样本未显示")
    else:
        print("\n🎉 所有预测都正确！")

def main():
    # 文件路径
    file_path = "/Users/cubicz/Documents/ustg/DSAA5020-Foundation of DSA/project/project/baseline/results-baseline-db.jsonl"  # 可以修改为你的文件路径
    
    # 处理数据
    total, correct, accuracy, results = process_jsonl_file(file_path)
    
    # 打印统计结果
    print("=" * 60)
    print("                模型准确率统计结果")
    print("=" * 60)
    print(f"总样本数: {total}")
    print(f"正确预测数: {correct}")
    print(f"错误预测数: {total - correct}")
    print(f"准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("=" * 60)
    
    # 打印详细结果（可选）
    print_detailed_results(results, max_display=20)
    
    # 保存结果到文件（可选）
    output_file = "analysis_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'total_samples': total,
            'correct_predictions': correct,
            'accuracy': accuracy,
            'results': results
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n详细结果已保存到 {output_file}")

if __name__ == "__main__":
    main()