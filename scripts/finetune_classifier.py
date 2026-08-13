import json, os, torch
from torch.utils.data import Dataset, Subset
from transformers import (
    Qwen2VLForConditionalGeneration, AutoProcessor,
    Trainer, TrainingArguments
)
from peft import LoraConfig, get_peft_model, TaskType
from PIL import Image

# ===== 配置 =====
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model/Qwen/Qwen2-VL-2B-Instruct")
DATA_PATH  = os.path.join(BASE_DIR, "data/ocr/labels.jsonl")
OUTPUT_DIR = os.path.join(BASE_DIR, "model/Qwen2-VL-2B-OCR")
LORA_R     = 16
EPOCHS     = 3
BATCH      = 4
LR         = 2e-4

# ===== 数据集 =====
class OCRDataset(Dataset):
    def __init__(self, data_path):
        with open(data_path) as f:
            self.data = [json.loads(line) for line in f if line.strip()]
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = os.path.join(os.path.dirname(DATA_PATH), item["image"])
        image = Image.open(img_path).convert("RGB")
        return {"image": image, "text": item["text"]}

# ===== 加载模型 =====
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, torch_dtype=torch.float16, local_files_only=True
)
for p in model.model.visual.parameters():
    p.requires_grad = False

lora = LoraConfig(
    r=LORA_R, lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.1, bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora)
model.print_trainable_parameters()

# ===== 数据 =====
processor = AutoProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
dataset = OCRDataset(DATA_PATH)

split = int(len(dataset) * 0.8)
train_data = Subset(dataset, range(split))
val_data   = Subset(dataset, range(split, len(dataset)))
print(f"训练集: {len(train_data)} 条, 验证集: {len(val_data)} 条")

def collate_fn(examples):
    texts, images = [], []
    for item in examples:
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": item["image"]},
                {"type": "text", "text": "识别图片中的所有文字，只输出文字内容，不要加任何解释。"}
            ]},
            {"role": "assistant", "content": item["text"]}
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False)
        texts.append(text)
        images.append(item["image"])

    inputs = processor(text=texts, images=images,
                       return_tensors="pt", padding=True)
    inputs["labels"] = inputs["input_ids"].clone()
    return inputs

# ===== 训练 =====
trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir=OUTPUT_DIR, num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH, learning_rate=LR,
        fp16=True, logging_steps=10, save_strategy="epoch",
        eval_strategy="epoch",
        report_to="none", remove_unused_columns=False,
    ),
    train_dataset=train_data,
    eval_dataset=val_data,
    data_collator=collate_fn,
)
trainer.train()

# ===== 合并并保存 =====
merged = model.merge_and_unload()
merged.save_pretrained(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
print(f"✅ 纯 OCR 模型保存到 {OUTPUT_DIR}")
