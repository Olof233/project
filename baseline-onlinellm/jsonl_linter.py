import json


def check_jsonl_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        total = 0
        custom_id_set = set()
        for line in file:
            if line.strip() == "":
                continue
            try:
                line_dict = json.loads(line)
            except json.decoder.JSONDecodeError:
                raise Exception(f"批量推理输入文件格式错误，第{total + 1}行非json数据")
            if not line_dict.get("custom_id"):
                raise Exception(f"批量推理输入文件格式错误，第{total + 1}行custom_id不存在")
            if not isinstance(line_dict.get("custom_id"), str):
                raise Exception(f"批量推理输入文件格式错误, 第{total + 1}行custom_id不是string")
            if line_dict.get("custom_id") in custom_id_set:
                raise Exception(
                    f"批量推理输入文件格式错误，custom_id={line_dict.get('custom_id', '')}存在重复"
                )
            else:
                custom_id_set.add(line_dict.get("custom_id"))
            if not isinstance(line_dict.get("body", ""), dict):
                raise Exception(
                    f"批量推理输入文件格式错误，custom_id={line_dict.get('custom_id', '')}的body非json字符串"
                )
            total += 1
    return total



if __name__ == "__main__":
    # 替换<YOUR_JSONL_FILE>为你的JSONL文件路径
    file_path = "batch_inference_input.jsonl"
    try:
        total_lines = check_jsonl_file(file_path)
        print(f"文件中有效JSON数据的行数为: {total_lines}")
    except FileNotFoundError:
        print(f"文件不存在: {file_path}")
    except Exception as e:
        print(f"检查出错: {e}")
