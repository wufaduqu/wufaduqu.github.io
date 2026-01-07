
import json
import os
import torch
import re # 引入正則表達式庫，用於後處理模型輸出
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel # 導入 PeftModel 用於載入 LoRA 權重

def generate_response(prompt_text: str, model, tokenizer, max_new_tokens: int = 256) -> str:
    """
    使用微調後的模型生成回應。

    用途: 根據使用者輸入的提示文本，生成模型的對話回應。
    預期輸入:
        prompt_text (str): 使用者輸入的提示文本。
        model: 已載入並設定為評估模式的模型 (PeftModel 或 AutoModelForCausalLM)。
        tokenizer: 已載入的 AutoTokenizer 物件。
        max_new_tokens (int, optional): 生成的最大新 token 數量。預設為 256。
    預期輸出:
        str: 模型生成的對話回應，經過後處理。
    潛在困難:
        1. 推理速度: 大型模型或較長的 `max_new_tokens` 會顯著增加生成時間。
           解決方案: 優化硬體、使用更小的模型、調整 `max_new_tokens`。
        2. 輸出清理準確性: 後處理邏輯可能無法完美處理所有模型輸出模式，特別是當模型生成格式不符合預期時。
           解決方案: 精煉後處理正則表達式，或調整訓練數據使其輸出更一致。
        3. 模型的生成品質: `temperature` 和 `top_p` 等參數會影響生成文本的隨機性和連貫性，需要適當調整。
    """
    # 格式化輸入為 Qwen 預期的對話格式
    # System Prompt: 設定模型的角色和行為。
    # User Prompt: 包含使用者實際輸入的內容。
    # Assistant Start Tag: 指示模型開始生成助手的回應。
    system_prompt = "你是名為「萊拉」的女僕助理。個性溫和、開朗、有禮貌，擅長打掃、整理家務與日常陪伴對話。回應時保持親切但專業，不涉及任何不當或越界內容。請直接給出萊拉的對話回應，不要包含任何思考過程、內部指令、計畫或以任何形式呈現的內心獨白，只需提供對話的最終答案。"
    formatted_input = f"<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{prompt_text}<|im_end|>
<|im_start|>assistant
"

    # 將格式化後的輸入文本轉換為 Token ID 序列
    input_ids = tokenizer.encode(formatted_input, return_tensors="pt")

    # 將輸入移到 GPU 上（如果可用），以加速推理
    if torch.cuda.is_available():
        input_ids = input_ids.cuda()

    with torch.no_grad(): # 在推理時禁用梯度計算，節省記憶體並加速
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens, # 最大生成 token 數量
            num_return_sequences=1,     # 只生成一個序列
            pad_token_id=tokenizer.pad_token_id, # 填充 token ID
            do_sample=True,             # 啟用抽樣生成，增加多樣性
            temperature=0.7,            # 調整生成溫度，較低值生成更保守，較高值生成更多樣
            top_p=0.9,                  # Top-p (nucleus) 抽樣，控制生成文本的品質和多樣性
            eos_token_id=tokenizer.eos_token_id # 結束符號 token ID
        )

    # 解碼生成的 Token ID 序列為文本
    # skip_special_tokens=False 以保留 <|im_end|> 等特殊標籤，便於後處理
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=False)

    # --- 後處理邏輯：提取助手回應並清理「思考模式」 --- #
    assistant_start_tag = '<|im_start|>assistant
'
    assistant_end_tag = '<|im_end|>'

    assistant_response = "無法解析助手回應。"
    assistant_start_index = decoded_output.find(assistant_start_tag)

    if assistant_start_index != -1:
        # 截取從助手開始標籤之後的內容
        raw_assistant_response = decoded_output[assistant_start_index + len(assistant_start_tag):]

        # 移除第一個 <|im_end|> 及其之後的所有內容，確保只獲取當前助手的回應
        assistant_end_index = raw_assistant_response.find(assistant_end_tag)
        if assistant_end_index != -1:
            raw_assistant_response = raw_assistant_response[:assistant_end_index]

        # 使用正則表達式清理模型可能生成的 <think>...</think> 內容
        # re.DOTALL 使得 '.' 可以匹配換行符，確保清理跨行的 <think> 標籤
        # 修正: 移除 \/ 中的反斜線，因為 '/' 不需要在正則表達式中轉義
        cleaned_response = re.sub(r'<think>.*?</think>', '', raw_assistant_response, flags=re.DOTALL).strip()

        # 進一步清理多餘的換行和空白字符，使其輸出更整潔
        # 修正: 替換 \s 為明確的空白字符集，避免 SyntaxWarning
        assistant_response = re.sub('[ 	
]{2,}', ' ', cleaned_response).strip() # 將多個空白替換為單個
        assistant_response = re.sub(r'
+', '
', assistant_response).strip() # 將多個換行符替換為單個

    return assistant_response

def main():
    """
    主函數，負責載入模型、Tokenizer 並提供互動式推理介面。

    用途: 整合模型載入和 `generate_response` 函數，提供一個可執行的推理流程。
    潛在困難:
        1. 模型載入錯誤: 路徑錯誤、文件損壞或資源不足（特別是 GPU 記憶體不足以載入基礎模型）。
           解決方案: 檢查路徑、確認文件完整性、調整 `BitsAndBytesConfig` 或使用較小的基礎模型。
        2. Tokenizer 配置困難: 如果 `pad_token` 或 `eos_token_id` 未正確設定，可能導致 Tokenization 或生成錯誤。
           解決方案: 參考 Qwen 官方文檔，手動設定這些特殊 Token。
    """
    # --- 1. 配置與初始化 --- #
    # 定義基礎模型名稱 (應與訓練時使用的基礎模型名稱一致)
    base_model_name = "Qwen/Qwen3-0.6B"

    # 微調後的 LoRA 權重路徑
    # 這應該指向保存 LoRA adapter 權重的資料夾，例如 './qwen_finetuned/final_qwen_lora_adapter'
    finetuned_model_path = "./qwen_finetuned/final_qwen_lora_adapter" 

    print(f"正在載入 Tokenizer for base model: {base_model_name}")
    # 載入 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    # Qwen 模型可能預設沒有 pad_token，需要手動添加
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'}) 
        print("Tokenizer 未設定 pad_token，已添加 '[PAD]' 作為填充符號。")
    # 設置 eos_token_id，對於 Qwen 模型，<|im_end|> 是其主要結束符號
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids('<|im_end|>')
        print(f"Tokenizer 的 eos_token_id 未設定，已設定為: {tokenizer.eos_token_id} (對應 '<|im_end|>')")

    # --- 2. 載入基礎模型 (使用 4-bit 量化) --- #
    # 為了保持與訓練時一致，並節省記憶體，我們同樣使用 4-bit 量化載入基礎模型。
    # 這對於在資源有限的環境中運行模型至關重要。
    bnb_config = None
    if torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        print("CUDA 可用，將使用 4-bit 量化載入基礎模型以節省記憶體。")
    else:
        print("CUDA 不可用，將以標準方式載入模型 (可能需要更多記憶體)。")

    print(f"正在載入基礎模型 '{base_model_name}'...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto" if torch.cuda.is_available() else None # 自動分配到可用的裝置
    )
    print("基礎模型載入完成。")

    # --- 3. 載入微調後的 LoRA 權重 --- #
    # LoRA (Low-Rank Adaptation) 權重是相對較小的權重矩陣，需要疊加到基礎模型上才能生效。
    # PeftModel.from_pretrained 函數會將這些 LoRA 權重加載到已有的基礎模型中，
    # 形成一個結合了基礎模型能力和微調知識的 "Adapter" 模型。
    # 這樣做的好處是，我們不需要保存和載入一個完整的、巨大的微調模型，只需保存和載入小的 LoRA 權重，極大地節省了儲存空間和載入時間。
    if os.path.exists(finetuned_model_path):
        print(f"從 '{finetuned_model_path}' 載入 LoRA 權重...")
        model_to_test = PeftModel.from_pretrained(base_model, finetuned_model_path)
        print("LoRA 權重載入成功。")
    else:
        print(f"錯誤：在 '{finetuned_model_path}' 中找不到 LoRA 權重。請確認路徑是否正確，或模型是否已成功微調並保存。")
        print("將使用基礎模型進行推理 (未微調)。")
        model_to_test = base_model

    # 將模型設定為評估模式。在評估模式下，模型會禁用 Dropout 等訓練特有的層，確保結果的一致性。
    model_to_test.eval()
    print("模型載入完成並設定為評估模式 (eval mode)。")

    # --- 4. 互動式推理迴圈 --- #
    print("
--- 開始互動式對話 ---")
    print("請輸入您的測試對話（輸入 'exit' 結束）：")
    while True:
        user_input = input("您：")
        if user_input.lower() == 'exit':
            break

        response = generate_response(user_input, model_to_test, tokenizer)
        print(f"模型：{response}")

    print("--- 對話結束 ---")

if __name__ == '__main__':
    # 確保當此檔案作為模組導入時，不會自動執行交互式對話，
    # 只有在直接運行此檔案時才會執行 `main()` 函數。
    main()
