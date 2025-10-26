# Fine-Tuning Guide: EmoGPT Dataset (1B-3B Models)

## 📋 Overview

This guide walks through fine-tuning small language models (1B-3B parameters) on the EmoGPT dataset to learn emoji-compressed conversational responses.

**Dataset:** 474 high-quality augmented conversations (32.5% token reduction)  
**Target Models:** 1B-3B parameter efficient models  
**Method:** QLoRA (Parameter-Efficient Fine-Tuning)  
**Goal:** Teach models to naturally produce emoji-compressed responses

---

## 🎯 Step 1: Prepare Your Dataset

### Current Format
Your dataset is in JSONL format with conversation history:

```json
{
  "messages": [
    {"role": "user", "content": "How do I install Python?"},
    {"role": "assistant", "content": "Download from python.org 💻..."}
  ],
  "metadata": {
    "tree_id": "...",
    "augmented": true,
    "original_tokens": 150,
    "augmented_tokens": 100,
    "token_reduction": 50,
    "reduction_pct": 33.33
  }
}
```

### Convert to Training Format

**Option A: Chat Template (Recommended for multi-turn)**

```python
# convert_to_training_format.py
import json
from datasets import Dataset

def convert_to_chat_format(input_file, output_file):
    """Convert JSONL to HuggingFace chat format"""
    data = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            conv = json.loads(line)
            
            # Only use successfully augmented conversations
            if conv['metadata'].get('augmented') == True:
                data.append({
                    'messages': conv['messages']
                })
    
    # Create HuggingFace dataset
    dataset = Dataset.from_list(data)
    
    # Split into train/validation (90/10)
    split = dataset.train_test_split(test_size=0.1, seed=42)
    
    # Save
    split['train'].to_json(f"{output_file}_train.jsonl")
    split['test'].to_json(f"{output_file}_val.jsonl")
    
    print(f"✅ Created {len(split['train'])} training samples")
    print(f"✅ Created {len(split['test'])} validation samples")

if __name__ == "__main__":
    convert_to_chat_format(
        'datasets/oasst_conversations_augmented.jsonl',
        'datasets/emogpt_train'
    )
```

**Run:**
```bash
python convert_to_training_format.py
```

**Output:**
- `datasets/emogpt_train_train.jsonl` (426 samples)
- `datasets/emogpt_train_val.jsonl` (48 samples)

---

## 🤖 Step 2: Select Your Model

### Recommended Models (1B-3B Range)

| Model | Size | Best For | Notes |
|-------|------|----------|-------|
| **Qwen2.5-1.5B-Instruct** | 1.5B | Best efficiency | Strong instruction following |
| **Phi-3.5-mini-instruct** | 3.8B | Quality | Microsoft's efficient architecture |
| **Gemma-2B-it** | 2B | Balance | Google's open model |
| **Llama-3.2-1B-Instruct** | 1B | Smallest viable | Meta's latest small model |
| **Llama-3.2-3B-Instruct** | 3B | Best baseline | Stronger performance |

**Recommendation:** Start with **Qwen2.5-1.5B-Instruct** or **Llama-3.2-3B-Instruct**

---

## 🛠️ Step 3: Set Up Training Environment

### Install Dependencies

```bash
# Create virtual environment
uv venv finetuning
source finetuning/bin/activate  # On Windows: finetuning\Scripts\activate

# Install packages
uv pip install torch transformers accelerate peft bitsandbytes datasets trl wandb
```

### Hardware Requirements

| Model Size | Minimum VRAM | Recommended | Training Time |
|------------|-------------|-------------|---------------|
| 1B | 6GB | 8GB | 2-3 hours |
| 1.5B | 8GB | 12GB | 3-4 hours |
| 3B | 12GB | 16GB | 4-6 hours |

**Can't meet requirements?** Use Google Colab (free T4 GPU) or Kaggle notebooks.

---

## 🚀 Step 4: Training Script (QLoRA)

### Why QLoRA?
- ✅ **Efficient:** 4-bit quantization reduces memory by 75%
- ✅ **Fast:** Trains in 2-4 hours on consumer GPUs
- ✅ **Quality:** Minimal performance degradation vs full fine-tuning
- ✅ **Small:** Adapter weights only (~50-100MB)

### Training Script

Create `train_emogpt.py`:

```python
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer
from datasets import load_dataset

# Configuration
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"  # Change to your chosen model
OUTPUT_DIR = "./emogpt-qwen-1.5b"
DATASET_PATH = "datasets/emogpt_train_train.jsonl"
VAL_DATASET_PATH = "datasets/emogpt_train_val.jsonl"

# QLoRA configuration (4-bit quantization)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Prepare model for k-bit training
model = prepare_model_for_kbit_training(model)

# LoRA configuration
lora_config = LoraConfig(
    r=16,                       # Rank (higher = more capacity, 8-64 typical)
    lora_alpha=32,             # Scaling factor (typically 2*r)
    target_modules=[           # Which layers to adapt
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,         # Dropout for regularization
    bias="none",
    task_type="CAUSAL_LM",
)

# Apply LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Load dataset
train_dataset = load_dataset('json', data_files=DATASET_PATH, split='train')
val_dataset = load_dataset('json', data_files=VAL_DATASET_PATH, split='train')

# Training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,              # 3-5 epochs for small datasets
    per_device_train_batch_size=4,   # Adjust based on VRAM
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,   # Effective batch size = 16
    gradient_checkpointing=True,     # Reduce memory usage
    learning_rate=2e-4,              # Higher than full fine-tuning
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=50,
    save_total_limit=2,
    fp16=True,                       # Mixed precision training
    optim="paged_adamw_8bit",       # Memory-efficient optimizer
    report_to="wandb",               # Change to "none" if no W&B
    run_name="emogpt-qwen-1.5b",
)

# Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    peft_config=lora_config,
    dataset_text_field="messages",
    max_seq_length=1024,
    tokenizer=tokenizer,
    args=training_args,
)

# Train
print("🚀 Starting training...")
trainer.train()

# Save final model
print("💾 Saving model...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"✅ Training complete! Model saved to {OUTPUT_DIR}")
```

### Run Training

```bash
# With W&B logging (recommended)
wandb login
python train_emogpt.py

# Without W&B
# Change report_to="none" in script, then:
python train_emogpt.py
```

---

## ⚙️ Step 5: Hyperparameter Tuning

### For Small Datasets (500 samples)

**Key Considerations:**
- ⚠️ **Overfitting risk:** Small dataset means model can memorize
- ✅ **More epochs OK:** 3-5 epochs (vs 1-2 for large datasets)
- ✅ **Higher learning rate:** 2e-4 (vs 1e-5 for large datasets)
- ✅ **Strong regularization:** Dropout, gradient clipping

### Recommended Settings

```python
# Conservative (less overfitting)
training_args = TrainingArguments(
    num_train_epochs=3,
    learning_rate=1e-4,
    weight_decay=0.01,
    warmup_ratio=0.1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,  # Effective batch = 16
)

lora_config = LoraConfig(
    r=8,              # Lower rank = less capacity
    lora_alpha=16,
    lora_dropout=0.1, # Higher dropout
)
```

```python
# Aggressive (better fit, risk overfitting)
training_args = TrainingArguments(
    num_train_epochs=5,
    learning_rate=2e-4,
    weight_decay=0.0,
    warmup_ratio=0.1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,  # Effective batch = 16
)

lora_config = LoraConfig(
    r=16,             # Higher rank = more capacity
    lora_alpha=32,
    lora_dropout=0.05, # Lower dropout
)
```

**Start with conservative, increase if underperforming.**

---

## 📊 Step 6: Monitor Training

### What to Watch

**1. Training Loss**
- Should decrease steadily
- Target: < 1.0 by end of training
- If stuck > 2.0: increase learning rate or epochs

**2. Validation Loss**
- Should decrease with training loss
- ⚠️ **Overfitting:** If train loss drops but val loss increases
- **Solution:** Early stopping, more dropout, lower rank

**3. Evaluation Metrics**
- Perplexity (lower is better)
- Token accuracy
- Manual sample inspection

### Example Good Training Curve

```
Epoch 1: train_loss=2.1, val_loss=2.0  ✅
Epoch 2: train_loss=1.5, val_loss=1.4  ✅
Epoch 3: train_loss=1.1, val_loss=1.0  ✅
```

### Example Overfitting

```
Epoch 1: train_loss=2.1, val_loss=2.0  ✅
Epoch 2: train_loss=1.5, val_loss=1.4  ✅
Epoch 3: train_loss=0.8, val_loss=1.6  ❌ Stop here!
```

---

## 🧪 Step 7: Test Your Model

### Inference Script

Create `test_emogpt.py`:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_PATH = "./emogpt-qwen-1.5b"

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

# Load LoRA adapter
model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model = model.merge_and_unload()  # Merge adapter for faster inference

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH, trust_remote_code=True)

# Test prompts
test_prompts = [
    "How do I install Python on my computer?",
    "Can you explain what machine learning is in simple terms?",
    "What's the best way to learn programming?",
    "I'm really happy with your help, thank you so much!"
]

print("🤖 Testing EmoGPT Model\n" + "="*60)

for prompt in test_prompts:
    # Format as chat
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Generate
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )
    
    # Decode
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    print(f"\n👤 User: {prompt}")
    print(f"🤖 Assistant: {response}")
    print("-"*60)
```

**Run:**
```bash
python test_emogpt.py
```

### Evaluation Checklist

- [ ] Responses use emojis appropriately (not too many, not too few)
- [ ] Technical accuracy preserved
- [ ] Shorter than typical LLM responses (check token count)
- [ ] Natural and helpful tone
- [ ] Emojis replace verbose phrases, not technical terms

---

## 🎯 Step 8: Evaluate Performance

### Quantitative Metrics

```python
# evaluate_model.py
import json
import tiktoken
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def evaluate_token_reduction(model, tokenizer, test_file):
    """Calculate average token reduction on test set"""
    encoding = tiktoken.get_encoding("cl100k_base")
    
    total_original = 0
    total_generated = 0
    
    with open(test_file, 'r') as f:
        for line in f:
            conv = json.loads(line)
            
            # Get original response
            original = conv['messages'][-1]['content']
            original_tokens = len(encoding.encode(original))
            
            # Generate model response
            messages = conv['messages'][:-1]  # Without last response
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.7)
            generated = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            generated_tokens = len(encoding.encode(generated))
            
            total_original += original_tokens
            total_generated += generated_tokens
    
    reduction = (total_original - total_generated) / total_original * 100
    print(f"Token Reduction: {reduction:.2f}%")
    return reduction

# Run evaluation
reduction = evaluate_token_reduction(model, tokenizer, 'datasets/emogpt_train_val.jsonl')
```

### Qualitative Review

**Sample 20-30 responses manually and check:**
1. Emoji usage (strategic vs decorative)
2. Technical accuracy
3. Readability and flow
4. Tone (helpful, warm, professional)

---

## 🚨 Common Issues & Solutions

### Issue: Model not using emojis

**Symptoms:** Output looks like normal LLM responses, no emojis

**Solutions:**
1. Increase training epochs (3 → 5)
2. Increase LoRA rank (8 → 16)
3. Lower learning rate (2e-4 → 1e-4) for more stable training
4. Check dataset: ensure augmented=true samples only

### Issue: Too many emojis (emoji spam)

**Symptoms:** Every sentence has multiple emojis

**Solutions:**
1. Review training data quality
2. Add few-shot examples in system prompt at inference
3. Reduce training epochs (overfitting)
4. Filter out over-augmented samples (>5 emojis per response)

### Issue: Model degraded (worse than base)

**Symptoms:** Incoherent responses, repetition, technical errors

**Solutions:**
1. Reduce learning rate (2e-4 → 1e-4)
2. Increase regularization (dropout 0.05 → 0.1)
3. Reduce epochs (5 → 3)
4. Check for data quality issues

### Issue: Out of memory

**Solutions:**
1. Reduce batch size (4 → 2 or 1)
2. Increase gradient accumulation (4 → 8)
3. Enable gradient checkpointing
4. Use smaller model (3B → 1.5B)

---

## 📈 Expected Results

### Realistic Targets for 474 Samples

| Metric | Target | Notes |
|--------|--------|-------|
| Token Reduction | 20-30% | Lower than prompted GPT (32.5%) |
| Training Loss | < 1.0 | By final epoch |
| Validation Loss | < 1.2 | Close to training loss |
| Emoji Appropriateness | 80%+ | Manual review |
| Technical Accuracy | 90%+ | No degradation from base |

**Why lower than 32.5%?**
- Small model (1-3B) vs large prompted model (120B)
- Small dataset (474 samples)
- First iteration

**Improvement path:**
- Add 1500-2500 LMSYS samples → 25-35% reduction
- Use larger model (7B) → 28-38% reduction

---

## 🔄 Next Steps After First Model

### 1. Expand Dataset
- Extract & augment LMSYS conversations
- Target: 2000-3000 total samples
- Re-train with expanded dataset

### 2. Try Larger Models
- Mistral-7B-Instruct (upper bound)
- Compare 1B vs 1.5B vs 3B vs 7B

### 3. Human Evaluation
- A/B testing with users
- Measure preference, warmth, helpfulness

### 4. Optimize
- Merge adapter into base model for faster inference
- Quantize to GGUF for CPU deployment
- Create API wrapper

---

## 📚 Resources

**Frameworks:**
- [HuggingFace PEFT](https://github.com/huggingface/peft)
- [TRL (Transformer Reinforcement Learning)](https://github.com/huggingface/trl)
- [Unsloth](https://github.com/unslothai/unsloth) (2x faster training)

**Tutorials:**
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [HuggingFace Fine-tuning Guide](https://huggingface.co/docs/transformers/training)
- [Weights & Biases Course](https://www.wandb.courses/courses/training-fine-tuning)

**Model Hubs:**
- [HuggingFace Models](https://huggingface.co/models)
- [Ollama](https://ollama.ai/) (local inference)

---

## ✅ Quick Start Checklist

- [ ] Dataset converted to chat format (train/val split)
- [ ] Dependencies installed (transformers, peft, trl, bitsandbytes)
- [ ] Model selected (Qwen2.5-1.5B or Llama-3.2-3B recommended)
- [ ] Training script configured
- [ ] GPU available (8GB+ VRAM) or cloud notebook ready
- [ ] W&B account created (optional but recommended)
- [ ] Run training (3-6 hours)
- [ ] Test model with inference script
- [ ] Evaluate token reduction and quality
- [ ] Iterate based on results

---

**Good luck with your fine-tuning! 🚀**

For questions or issues, refer to the documentation in `datapipeline/README.md` or check the research plan in `newplan.md`.

