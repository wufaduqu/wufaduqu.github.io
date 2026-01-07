# Qwen/Qwen3-0.6B 角色聊天機器人微調專案

## 專案概述

此專案旨在透過微調 Qwen/Qwen3-0.6B 模型，建立一個具有特定角色（例如：溫和、開朗、有禮貌的女僕助理「萊拉」）的聊天機器人。我們將使用參數高效微調 (PEFT) 技術，特別是 LoRA，來最小化計算資源的需求，並利用 `bitsandbytes` 進行 4-bit 量化以進一步節省記憶體。專案提供了從資料準備、模型訓練到推理測試的完整流程。

## 環境設定

為了確保專案的可重現性，我們提供了一個 `requirements.txt` 檔案，其中列出了所有必要的 Python 套件及其版本。

### 1. 安裝套件

在您的環境中，首先確保您有 `pip`。然後，您可以透過以下指令安裝所有依賴：

```bash
pip install -r requirements.txt
```

`requirements.txt` 檔案內容範例如下：
```
transformers==4.57.3
torch==2.9.0+cu126
accelerate==1.12.0
peft==0.18.0
bitsandbytes==0.49.0
datasets==4.0.0
```

### 2. `torch` 版本與 CUDA/CuDNN 相容性考量

特別要注意 `torch==2.9.0+cu126` 這一行。這表示 `torch` 是為 CUDA 12.6 編譯的版本。如果您有 NVIDIA GPU，請確保您的 CUDA/CuDNN 版本與此相符，以充份利用 GPU 加速。

*   **無 GPU 或 CUDA 版本不符**：如果您的環境沒有 NVIDIA GPU，或 CUDA 版本不同，直接安裝此版本可能會失敗或無法利用 GPU。在這種情況下，**強烈建議訪問 PyTorch 官方網站 (`https://pytorch.org/get-started/locally/`)**，根據您的作業系統、Python 版本和 CUDA 版本（或選擇 CPU Only 版本）獲取精確的安裝指令。
*   **`bitsandbytes`**：此套件用於模型的 4-bit 量化，可以顯著減少 GPU 記憶體使用。它需要 CUDA 環境才能正常工作。

## 資料準備

資料準備是微調模型的關鍵步驟。我們需要將您的對話資料轉換為模型可訓練的特定格式。`data_processing.py` 檔案提供了必要的工具。

### 1. 所需資料格式

模型期望的輸入是一個 JSON Lines (`.jsonl`) 檔案，其中每行是一個 JSON 物件，包含一個 `id` 和一個 `conversations` 列表。`conversations` 列表則是由 `{ "from": "role", "value": "content" }` 形式的字典組成的多輪對話。

例如：
```json
{"id": "conv_1", "conversations": [{"from": "system", "value": "系統提示"}, {"from": "user", "value": "使用者輸入"}, {"from": "assistant", "value": "助手回應"}]}
```

### 2. `data_processing.py` 中的關鍵函數

*   **`convert_txt_to_jsonl(txt_file_path, jsonl_file_path)`**：
    *   **用途**：將特定格式的 `.txt` 檔案轉換為 `.jsonl`。預期 `.txt` 檔案為每兩行一組，第一行為 prompt，第二行為 completion。
    *   **範例**：`python -c "import data_processing; data_processing.convert_txt_to_jsonl('your_input.txt', 'output_data.jsonl')"`
*   **`load_jsonl_dataset(file_path)`**：
    *   **用途**：載入 `.jsonl` 檔案，並將其內容解析為 Python 物件列表。
    *   **範例**：`loaded_data = data_processing.load_jsonl_dataset('output_data.jsonl')`
*   **`format_qwen_conversation(conversation_list)`**：
    *   **用途**：將標準化的對話列表格式化為 Qwen 模型訓練所需的特定字串格式 (`<|im_start|>role
content<|im_end|>
`)。
*   **`tokenize_dataset(hf_dataset, tokenizer, max_length)`**：
    *   **用途**：對 Hugging Face `Dataset` 物件進行 Tokenization，並為因果語言模型準備標籤。將文本數據轉換為數值型 Token ID 序列。

### 3. 資料準備範例指令

1.  **若您的資料是特定 `.txt` 格式**：
    ```bash
    python -c "import data_processing; data_processing.convert_txt_to_jsonl('your_input.txt', 'output_data.jsonl')"
    ```
2.  **若您的資料已是 `.jsonl` 格式**：確保其符合上述提及的對話結構。


## 訓練模型指南

`train_model.py` 檔案包含了模型微調的完整邏輯。

### 1. 執行訓練

要開始訓練，請執行：

```bash
python train_model.py
```

### 2. 主要功能與配置

*   **模型載入與 4-bit 量化**：程式碼使用 `transformers.AutoModelForCausalLM` 載入 Qwen/Qwen3-0.6B 基礎模型，並透過 `BitsAndBytesConfig` 啟用 4-bit 量化。這顯著減少了 GPU 記憶體需求，使得在有限資源下也能訓練模型。
    *   `load_in_4bit=True`, `bnb_4bit_use_double_quant=True`, `bnb_4bit_quant_type="nf4"`, `bnb_4bit_compute_dtype=torch.bfloat16` 是建議的量化參數。
*   **PEFT (LoRA) 配置**：使用 `peft.LoraConfig` 配置 LoRA 參數。`r` (秩) 和 `lora_alpha` 決定了 LoRA 矩陣的規模和縮放因子。`target_modules` 則指定了哪些層應用 LoRA（例如：`["q_proj", "v_proj"]`）。這使得我們只訓練少量參數，而非整個模型，極大地提高了訓練效率。
*   **`TrainingArguments` 設定**：
    *   `output_dir`: 模型檢查點和日誌的保存路徑。
    *   `per_device_train_batch_size` 和 `gradient_accumulation_steps`：這些參數共同影響訓練的有效批次大小。`gradient_accumulation_steps` 在 GPU 記憶體有限時特別有用，它允許累計多個小批次的梯度來模擬更大的批次，從而提高訓練穩定性和效率。
    *   `num_train_epochs`, `learning_rate`, `logging_steps`, `save_steps` 等：控制訓練的迭代次數、學習率、日誌記錄頻率和檢查點保存頻率。
*   **`Trainer` 建立與訓練流程**：`transformers.Trainer` 類別被用於簡化訓練迴圈的管理，包括優化器、排程器、日誌記錄和檢查點保存。

### 3. 接續訓練 (Resume Training)

如果需要從上次中斷的地方繼續訓練，可以修改 `train_model.py` 中的 `resume_from_checkpoint_path` 變數為您要接續的檢查點路徑（例如：`./qwen_finetuned/checkpoint-186`）。

## 推理測試說明

`inference.py` 檔案用於載入微調後的模型並進行互動式推理測試。

### 1. 執行推理

要啟動互動式推理介面，請執行：

```bash
python inference.py
```

### 2. 主要功能

*   **模型載入**：程式碼會載入 Qwen/Qwen3-0.6B 基礎模型，並疊加微調後的 LoRA 權重。與訓練時一樣，模型會使用 4-bit 量化載入，以節省記憶體。
*   **`generate_response(prompt_text, model, tokenizer)` 函數**：
    *   **輸入提示詞格式**：該函數會將您的提示詞與預設的 `system_prompt` 結合，格式化為 Qwen 模型理解的對話格式 (`<|im_start|>system
...<|im_end|>
<|im_start|>user
...<|im_end|>
<|im_start|>assistant
`)。
    *   **輸出後處理**：模型生成回應後，`generate_response` 函數會進行後處理，以提取助手的回應部分，並使用正則表達式移除模型可能生成的任何內部「思考模式」標籤（例如 `<think>...</think>`），確保輸出乾淨整潔。
*   **互動式測試**：運行 `inference.py` 後，您將進入一個迴圈，可以不斷輸入問題並查看模型的回應，直到輸入 `exit` 結束。

### 3. 互動式測試範例

```
--- 開始互動式對話 ---
請輸入您的測試對話（輸入 'exit' 結束）：
您：萊拉，今天家裡好像有點亂，你可以幫我想想從哪裡開始整理嗎？
模型：好的，主人。萊拉建議可以先從最常使用的客廳開始整理喔。先把桌面與地板上的物品分類歸位，再進行簡單的擦拭與掃地，這樣整體空間會立刻清爽許多。
您：我今天有點累，不太想動。
模型：辛苦了呢，主人。那不如先好好休息一下吧。萊拉可以陪你聊聊天，或者幫你規劃一個輕鬆的小整理清單，等精神好一點再慢慢完成也沒關係。
您：exit
--- 對話結束 ---
```

## 除錯與重現結果

### 1. 常見問題

*   **`FileNotFoundError`**：檢查資料檔案（例如 `output_data.jsonl`）和模型檢查點路徑是否正確。
*   **`json.JSONDecodeError`**：您的 `.jsonl` 檔案中可能包含格式不正確的 JSON 行。請檢查檔案內容。
*   **CUDA 記憶體不足**：
    *   降低 `train_model.py` 中 `per_device_train_batch_size` 的值。
    *   增加 `gradient_accumulation_steps` 的值。
    *   降低 `max_length` 參數。
    *   確保 `bitsandbytes` 已正確安裝並啟用 4-bit 量化。
*   **`torch` 版本不相容**：參閱上述「環境設定」中的 `torch` 與 CUDA 相容性說明，確保安裝了適合您環境的 PyTorch 版本。
*   **`Tokenizer` 錯誤（例如沒有 `pad_token` 或 `eos_token_id`）**：程式碼中已包含自動添加 `[PAD]` 作為 `pad_token` 和 `<|im_end|>` 作為 `eos_token_id` 的邏輯，但如果遇到其他 Tokenizer 相關錯誤，請檢查 Tokenizer 的設定是否與 Qwen 模型相符。
*   **模型輸出異常**：
    *   檢查訓練資料是否足夠多樣和清潔。
    *   調整 `generate_response` 函數中的 `temperature` 和 `top_p` 參數，以調整生成的多樣性。
    *   檢查 `system_prompt` 是否清晰有效。

### 2. 重現結果的建議

1.  **環境一致性**：使用 `requirements.txt` 確保所有依賴套件版本完全一致。
2.  **資料一致性**：確保使用的訓練資料集完全相同。
3.  **隨機種子**：在訓練程式碼中設置所有可能的隨機種子（`torch.manual_seed()`, `numpy.random.seed()`, `random.seed()` 等），以減少結果的隨機性。
4.  **檢查點管理**：妥善保存和管理訓練過程中的檢查點，以便從特定狀態恢復或進行比較分析。

希望這份 `README.md` 能幫助您順利使用本專案！