
import json
import os
from transformers import AutoTokenizer
from datasets import Dataset

def convert_txt_to_jsonl(txt_file_path: str, jsonl_file_path: str):
    """
    將 .txt 檔案轉換為 .jsonl 格式。
    此函數預期 .txt 檔案的內容為每兩行一組，第一行為提示 (prompt)，第二行為回應 (completion)。
    轉換後為 {'prompt': '...', 'completion': '...'} 的 JSON Lines 格式。

    用途: 處理原始文字檔，將其結構化為機器學習模型可用的 JSON Lines 格式數據。
    預期輸入:
        txt_file_path (str): 輸入 .txt 檔案的路徑。
        jsonl_file_path (str): 輸出 .jsonl 檔案的路徑。
    預期輸出:
        成功轉換後，會在指定路徑生成一個 .jsonl 檔案。
    潛在困難:
        1. 原始 .txt 檔案格式不符合預期（例如，不是每兩行一組），導致解析錯誤。
        2. 編碼問題（例如 UTF-8 vs Big5），可能需要指定 encoding 參數。
        3. 檔案讀寫權限問題。
    """
    converted_data = []
    try:
        with open(txt_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            # 確保有偶數行或處理最後一行可能不完整的情況
            for i in range(0, len(lines), 2):
                if i + 1 < len(lines):
                    prompt = lines[i].strip()
                    completion = lines[i+1].strip()
                    if prompt and completion: # 確保 prompt 和 completion 都不為空
                        converted_data.append({"prompt": prompt, "completion": completion})
                    else:
                        print(f"警告：在檔案 '{txt_file_path}' 的第 {i+1}-{i+2} 行發現空行或不完整的對話，該組將被跳過。")
                else:
                    print(f"警告：在檔案 '{txt_file_path}' 的第 {i+1} 行後發現不完整的對話組，該行將被跳過。")

        with open(jsonl_file_path, 'w', encoding='utf-8') as f_out:
            for entry in converted_data:
                f_out.write(json.dumps(entry, ensure_ascii=False) + '
')

        print(f"成功將 '{txt_file_path}' 轉換為 '{jsonl_file_path}'，共 {len(converted_data)} 條資料。")
    except FileNotFoundError:
        print(f"錯誤：找不到檔案 '{txt_file_path}'。請確認檔案路徑是否正確。")
    except Exception as e:
        print(f"轉換檔案時發生錯誤：{e}")

def load_jsonl_dataset(file_path: str) -> list:
    """
    載入 .jsonl 檔案，並將其內容解析為 Python 物件列表。

    用途: 從 .jsonl 檔案中讀取結構化數據，為後續處理做準備。
    預期輸入:
        file_path (str): 輸入 .jsonl 檔案的路徑。
    預期輸出:
        list: 包含字典的列表，每個字典代表一行 JSON 數據。
    潛在困難:
        1. 檔案不存在 (FileNotFoundError)。
        2. .jsonl 檔案中某行不是有效的 JSON 格式 (json.JSONDecodeError)。
        3. 記憶體限制（如果檔案過大一次性載入，可能需要迭代讀取）。
    """
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line: # 跳過空行
                    continue
                try:
                    json_obj = json.loads(line)
                    data.append(json_obj)
                except json.JSONDecodeError as e:
                    print(f"警告：在檔案 '{file_path}' 的第 {line_num} 行發現 JSON 格式錯誤，該行將被跳過。")
                    print(f"錯誤訊息：{e}")
                    print(f"有問題的行內容：{line}")
                    print("-" * 20)
    except FileNotFoundError:
        print(f"錯誤：找不到檔案 '{file_path}'。請確認檔案路徑是否正確。")
    except Exception as e:
        print(f"載入 .jsonl 檔案時發生錯誤：{e}")
    return data

def format_qwen_conversation(conversation_list: list) -> str:
    """
    將標準化的對話列表格式化為 Qwen 模型訓練所需的特定字串格式。
    格式為 `<|im_start|>role
content<|im_end|>
`。

    用途: 將對話數據轉換為模型能理解的輸入序列格式。
    預期輸入:
        conversation_list (list): 包含對話回合字典的列表，例如
                                  `[{'from': 'user', 'value': '...'}, {'from': 'assistant', 'value': '...'}]`。
    預期輸出:
        str: 格式化後的單一字串，代表一個完整的 Qwen 對話序列。
    潛在困難:
        1. 輸入的 `conversation_list` 格式不正確（例如，缺少 'from' 或 'value' 鍵）。
        2. 特殊字元處理。
        3. 模型預期格式變更，可能需要修改此函數的輸出格式。
    """
    formatted_string = ""
    for turn in conversation_list:
        role = turn.get("from")
        content = turn.get("value")
        if role and content is not None:
            formatted_string += f"<|im_start|>{role}
{content}<|im_end|>
"
        else:
            print(f"警告：對話回合中缺少 'from' 或 'value' 鍵，跳過該回合: {turn}")
    return formatted_string

def tokenize_dataset(hf_dataset: Dataset, tokenizer: AutoTokenizer, max_length: int = 512) -> Dataset:
    """
    對 Hugging Face `Dataset` 物件進行 Tokenization，並為因果語言模型準備標籤。

    用途: 將文本數據轉換為模型輸入所需的數值型 Token ID 序列。
    預期輸入:
        hf_dataset (datasets.Dataset): 包含 'text' 欄位的 Hugging Face Dataset 物件。
        tokenizer (transformers.AutoTokenizer): 已初始化的 AutoTokenizer 物件。
        max_length (int, optional): 最大序列長度。預設為 512。
    預期輸出:
        datasets.Dataset: Tokenization 後的 Hugging Face Dataset 物件，包含 'input_ids'、'attention_mask' 和 'labels'。
    潛在困難:
        1. Tokenizer 配置不正確（例如，`pad_token` 未設定，導致填充錯誤）。
           通常需要確保 `tokenizer.pad_token` 已設定，若無則添加。
        2. `max_length` 過短導致重要信息截斷。
        3. Tokenization 速度慢（對於大型數據集）。
        4. 特殊 Token（如 EOS/BOS）的處理，需要確保它們被正確識別和使用。
    """
    if tokenizer.pad_token is None:
        # Qwen 模型可能預設沒有 pad_token，需要手動添加
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        print("警告：Tokenizer 未設定 pad_token，已添加 '[PAD]' 作為填充符號。")

    def _tokenize_function(examples):
        # 使用 tokenizer 對文本進行編碼
        # padding='max_length' 會將序列填充到最大長度
        # truncation=True 會截斷超過最大長度的序列
        tokenized_inputs = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_length)
        tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy() # 對於因果語言模型，標籤通常是輸入 ID 的副本
        return tokenized_inputs

    # 移除原始的 'text' 欄位，以避免與 tokenized_inputs 中的 'input_ids' 等衝突
    tokenized_dataset = hf_dataset.map(_tokenize_function, batched=True, remove_columns=hf_dataset.column_names)
    return tokenized_dataset

if __name__ == '__main__':
    print("這是一個用於 Qwen 模型資料處理的模組。")
    print("它包含以下功能：")
    print("1. `convert_txt_to_jsonl(txt_file_path, jsonl_file_path)`: 將特定格式的 .txt 轉換為 .jsonl。")
    print("2. `load_jsonl_dataset(file_path)`: 載入 .jsonl 檔案。")
    print("3. `format_qwen_conversation(conversation_list)`: 將對話格式化為 Qwen 模型所需的輸入字串。")
    print("4. `tokenize_dataset(hf_dataset, tokenizer, max_length)`: 對 Hugging Face Dataset 進行 Tokenization。")
    print("
以下是一些範例用法（已註釋掉，不會自動執行）：")

    # # 範例 1: 將 .txt 轉換為 .jsonl
    # # 假設存在一個 input.txt，內容為：
    # # 你好
    # # 你好，很高興為你服務。
    # # 今天天氣真好
    # # 是的，非常適合出門走走呢！
    # # convert_txt_to_jsonl('input.txt', 'output.jsonl')

    # # 範例 2: 載入 .jsonl 檔案
    # # dataset_list = load_jsonl_dataset('output.jsonl')
    # # print(f"載入的資料樣本: {dataset_list[:2]}")

    # # 範例 3: 格式化 Qwen 對話
    # # sample_conversation = [
    # #     {"from": "system", "value": "你是一個AI助手。"},
    # #     {"from": "user", "value": "你好嗎？"},
    # #     {"from": "assistant", "value": "我很好，謝謝。"}
    # # ]
    # # formatted_text = format_qwen_conversation(sample_conversation)
    # # print(f"格式化後的對話: {formatted_text}")

    # # 範例 4: Tokenization Dataset
    # # 假設 dataset_list 已經載入，並且每個字典有一個 'conversations' 鍵
    # # 且 'conversations' 鍵的值是一個對話列表
    # # 這裡為了演示，我們手動創建一個包含 'text' 欄位的 Hugging Face Dataset
    # # model_name = "Qwen/Qwen3-0.6B"
    # # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # # # Qwen models don't have a pad token by default, add one
    # # if tokenizer.pad_token is None:
    # #     tokenizer.add_special_tokens({'pad_token': '[PAD]'}) # 使用 add_special_tokens 而不是直接賦值

    # # sample_data = [
    # #     {"id": "conv_1", "conversations": [{"from": "system", "value": "角色設定"}, {"from": "user", "value": "使用者問"}, {"from": "assistant", "value": "助手答"}]},
    # #     {"id": "conv_2", "conversations": [{"from": "system", "value": "另一個角色"}, {"from": "user", "value": "另一使用者問"}, {"from": "assistant", "value": "另一助手答"}]}
    # # ]
    # # hf_dataset = Dataset.from_list(sample_data)

    # # def _format_for_tokenize(examples):
    # #     formatted_texts = []
    # #     for conv_list in examples['conversations']:
    # #         formatted_texts.append(format_qwen_conversation(conv_list))
    # #     return {"text": formatted_texts}
    # # hf_dataset_formatted = hf_dataset.map(_format_for_tokenize, batched=True)

    # # tokenized_hf_dataset = tokenize_dataset(hf_dataset_formatted, tokenizer)
    # # print(f"Tokenized Dataset 樣本: {tokenized_hf_dataset[0]}")
