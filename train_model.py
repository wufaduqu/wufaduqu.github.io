
import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from datasets import Dataset

# 引入我們自定義的資料處理模組
import data_processing

def main():
    # --- 1. 配置與初始化 --- #
    # 定義要使用的基礎模型名稱
    model_name = "Qwen/Qwen3-0.6B"
    # 定義資料檔案路徑 (假設已由 data_processing.py 轉換生成)
    data_file_path = 'output_data.jsonl' # 請確保這個檔案存在且格式正確
    # 定義微調模型的輸出目錄
    output_dir = "./qwen_finetuned"

    # 如果要從檢查點接續訓練，請設定此路徑。第一次訓練請設定為 None。
    # 範例: resume_from_checkpoint_path = "./qwen_finetuned/checkpoint-186"
    resume_from_checkpoint_path = None # 或者指定一個有效的檢查點路徑

    print(f"正在載入 Tokenizer for model: {model_name}")
    # 載入 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Qwen 模型可能預設沒有 pad_token，需要手動添加，否則 Tokenization 會報錯
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'}) # 使用 add_special_tokens 而不是直接賦值
        print("Tokenizer 未設定 pad_token，已添加 '[PAD]' 作為填充符號。")
    # 設置 eos_token_id，對於某些模型，這可能需要手動設置
    if tokenizer.eos_token_id is None:
        # Qwen 模型的結束符號通常是 <|im_end|>
        tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids('<|im_end|>')
        print(f"Tokenizer 的 eos_token_id 未設定，已設定為: {tokenizer.eos_token_id} (對應 '<|im_end|>')")

    # --- 2. 資料準備 --- #
    print(f"正在載入資料檔案 '{data_file_path}'...")
    # 載入 JSON Lines 格式的資料
    loaded_data = data_processing.load_jsonl_dataset(data_file_path)
    if not loaded_data:
        print("錯誤：未能載入任何資料，請檢查資料檔案和路徑。")
        return

    # 將載入的資料轉換為 Hugging Face Dataset 格式
    hf_dataset = Dataset.from_list(loaded_data)
    print(f"成功載入 {len(hf_dataset)} 條資料。")

    print("正在格式化資料為 Qwen 模型訓練所需的對話格式...")
    # 應用 format_qwen_conversation 函數來格式化每個對話樣本
    # 這裡假設 hf_dataset 中的每個元素都有一個 'conversations' 鍵，其值是一個對話列表
    def _format_for_tokenize(examples):
        formatted_texts = []
        for conv_list in examples['conversations']:
            formatted_texts.append(data_processing.format_qwen_conversation(conv_list))
        return {"text": formatted_texts}

    hf_dataset_formatted = hf_dataset.map(_format_for_tokenize, batched=True, remove_columns=['id', 'conversations']) # 移除原始欄位，僅保留 'text'
    print("資料格式化完成。")
    print(f"格式化後資料集範例 (第一條): {hf_dataset_formatted['text'][0][:200]}...")

    print("正在對資料集進行 Tokenization...")
    # 對格式化後的資料集進行 Tokenization
    # max_length: 決定了每個序列的最大長度。過短會截斷重要資訊，過長會增加記憶體消耗和計算量。
    # 對於 Qwen-0.6B 這種小型模型，512 是一個常見且合理的起點。應根據資料的實際長度調整。
    tokenized_dataset = data_processing.tokenize_dataset(hf_dataset_formatted, tokenizer, max_length=512)
    print("資料 Tokenization 完成。")
    print(f"Tokenized 資料集範例 (第一條input_ids): {tokenized_dataset['input_ids'][0][:50]}...")

    # 劃分訓練集和驗證集 (可選，對於小資料集可以直接全部用於訓練)
    # 如果資料集夠大，建議分割，以便在訓練過程中監控模型的泛化能力。例如 train_test_split(test_size=0.1, seed=42)。
    # 為了簡化範例，這裡直接使用整個 Tokenized 資料集作為訓練集。
    train_dataset = tokenized_dataset # 為簡化，使用整個資料集作為訓練集
    eval_dataset = None # 目前沒有設定驗證集

    # --- 3. 模型載入與量化配置 --- #
    # BitsAndBytesConfig 用於 4-bit 量化配置：極大地減少模型的記憶體佔用，
    # 使得在資源有限的 GPU 上也能載入和訓練大型模型。
    # load_in_4bit=True: 啟用 4-bit 量化。
    # bnb_4bit_use_double_quant=True: 啟用雙重量化，進一步節省記憶體。
    # bnb_4bit_quant_type="nf4": 使用 NF4 格式進行量化，通常比 FP4 效果好。
    # bnb_4bit_compute_dtype=torch.bfloat16: 量化後的計算數據類型，bfloat16 在保持精度與節省記憶體之間取得良好平衡。
    #   這對於 Qwen 模型來說很重要，因為它們通常使用 bfloat16 進行訓練。
    bnb_config = None
    if torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        print("CUDA 可用，將使用 4-bit 量化載入模型以節省記憶體。")
    else:
        print("CUDA 不可用，將以標準方式載入模型 (可能需要更多記憶體)。")
        print("警告：若無 GPU 支援，4-bit 量化可能無法完全發揮作用或訓練過程會非常慢。")

    print(f"正在載入基礎模型: {model_name}...")
    # 載入基礎模型
    # device_map="auto": 自動將模型層分配到可用的裝置（如多個 GPU 或 CPU）。對於量化模型通常是必須的。
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto" if torch.cuda.is_available() else None # 自動分配到可用的裝置
    )
    print("基礎模型載入完成。")

    # --- 4. PEFT (LoRA) 配置與應用 --- #
    # LoRA (Low-Rank Adaptation) 配置：一種參數高效微調技術，只訓練少量新增的低秩矩陣，
    # 大幅減少了微調所需的計算量和記憶體，同時能維持模型性能。
    # r (rank): LoRA 矩陣的秩。較大的 r 通常意味著更強的表達能力，但也增加參數量。8 或 16 是常見選擇。
    # lora_alpha: LoRA 的縮放因子。通常設為 r 的兩倍或等於 r。
    # target_modules: 應用 LoRA 的目標模組名稱。對於 Qwen 模型，通常是 Attention 層的線性投影（q_proj, k_proj, v_proj, o_proj）
    # 以及 MLP 層的線性投影（gate_proj, up_proj, down_proj）。正確選擇這些模組對性能至關重要，
    # 可以通過檢查模型的結構來確定 (例如 model.modules)。
    # 這裡的 'q_proj', 'v_proj' 是 Qwen 模型中常見且有效的 LoRA 注入點。根據模型架構，可能需要更多模組。
    # 可透過 `model.print_trainable_parameters()` 來確認 LoRA 是否成功應用以及訓練參數的數量。
    lora_config = LoraConfig(
        r=8, 
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"], # Qwen 模型常用的 LoRA 目標模組
        # 建議可以嘗試以下更全面的模組列表，效果可能更好，但訓練參數也會增加：
        # target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    print("LoRA 配置完成。")

    # 將 LoRA 配置應用到基礎模型上，生成可訓練的 PEFT 模型
    model = get_peft_model(model, lora_config)
    print("LoRA 模型創建成功。可訓練參數概覽：")
    model.print_trainable_parameters()
    # 潛在困難：如果 target_modules 選擇不當，可能導致訓練效果不佳。
    # 需要根據具體模型架構進行驗證和調整。

    # --- 5. 訓練參數設定 --- #
    # TrainingArguments 定義了訓練過程中的所有超參數和行為。
    # output_dir: 保存模型檢查點和日誌的目錄。
    # per_device_train_batch_size: 每個 GPU (或 CPU) 上的訓練批次大小。這對 GPU 記憶體消耗影響最大。
    # gradient_accumulation_steps: 梯度累計步數。當 GPU 記憶體不足以容納大的 batch_size 時，
    # 可以通過累計多個小 batch 的梯度來模擬大 batch，有效增加總 batch_size，同時節省記憶體。
    # 實際有效 batch_size = per_device_train_batch_size * gradient_accumulation_steps * num_gpus (如果有多 GPU)。
    # learning_rate: 學習率，微調模型的重要超參數。通常比預訓練時小。
    # num_train_epochs: 訓練的總輪數。需要根據資料集大小和模型收斂情況調整。
    # logging_dir: 日誌保存目錄，用於 TensorBoard 等。
    # logging_steps: 每隔多少步記錄一次日誌，便於監控訓練進度。
    # save_steps: 每隔多少步保存一次模型檢查點，以防訓練中斷或用於接續訓練。
    # save_total_limit: 最多保存的檢查點數量，防止佔用過多磁碟空間。
    # push_to_hub: 是否將模型推送到 Hugging Face Hub (需要登入並配置權限)。
    # report_to: 日誌報告平台，如 "wandb", "tensorboard" 或 "none"。
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2, # 根據 GPU 記憶體調整。較小的值可能需要配合 gradient_accumulation_steps。
        gradient_accumulation_steps=4, # 模擬更大的 batch size (2 * 4 = 8)。
        learning_rate=2e-4,
        num_train_epochs=3,
        logging_dir="./logs",
        logging_steps=10,
        save_steps=50,
        save_total_limit=2,
        push_to_hub=False,
        report_to="none"
    )
    print("訓練參數設定完成。")

    # --- 6. 建立 Trainer --- #
    # Trainer 類是 Hugging Face Transformers 提供的高級 API，用於簡化模型訓練過程。
    # 它處理了訓練迴圈、評估、日誌記錄、檢查點保存等複雜任務。
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset # 如果有驗證集，可以傳入
    )
    print("Trainer 建立完成。")

    # --- 7. 開始訓練 --- #
    print("開始微調模型...")
    # trainer.train() 會啟動訓練過程。
    # resume_from_checkpoint 參數用於從指定路徑的檢查點接續訓練。
    # 檢查點路徑應該指向一個包含 `trainer_state.json` 和模型權重的目錄，例如 `./output_dir/checkpoint-XXX`。
    # 這對於長時間訓練或中途意外中斷後恢復訓練非常有用。
    try:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint_path)
        print("模型微調完成！")

        # --- 8. 保存最終模型 --- #
        # 訓練完成後，保存 LoRA adapter 的權重。
        # 這些權重通常儲存在一個名為 'adapter_model.safetensors' 或 'adapter_model.bin' 的檔案中。
        # 它們比完整的基礎模型小得多，方便部署。
        final_adapter_output_path = os.path.join(output_dir, "final_qwen_lora_adapter")
        trainer.save_model(final_adapter_output_path)
        print(f"最終 LoRA adapter 已保存到 '{final_adapter_output_path}'")

    except Exception as e:
        print(f"訓練過程中發生錯誤：{e}")
        # 在這裡可以添加更多的錯誤處理邏輯，例如記錄詳細錯誤日誌等。


if __name__ == '__main__':
    # 在 `if __name__ == '__main__':` 區塊中運行主要訓練邏輯
    # 這樣確保當此檔案被作為模組導入時，訓練代碼不會自動執行，而只在直接運行時執行。
    # 所有的設置、數據加載、模型訓練都在此函數內完成，使其自包含且易於調用。
    # 避免在此處放置交互式測試代碼，以免影響自動化訓練流程。
    main()
