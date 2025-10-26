"""
Fine-tune Qwen2.5-1.5B-Instruct on EmoGPT dataset using Unsloth.
Optimized for A6000 48GB VRAM with slightly aggressive hyperparameters.
"""

import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_NAME = "unsloth/Qwen2.5-1.5B-Instruct"  # Unsloth optimized version
MAX_SEQ_LENGTH = 2048  # Qwen2.5 supports up to 32k, but 2k is sufficient
LOAD_IN_4BIT = True  # 4-bit quantization for efficiency

OUTPUT_DIR = "./emogpt-qwen-1.5b"
TRAIN_DATA = "emogpt_train.jsonl"
VAL_DATA = "emogpt_val.jsonl"

# Slightly aggressive hyperparameters for small dataset
EPOCHS = 5
BATCH_SIZE = 8  # With 48GB VRAM, we can afford larger batches
GRAD_ACCUM = 2  # Effective batch size = 16
LEARNING_RATE = 2e-4
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01

# LoRA configuration - slightly aggressive
LORA_R = 16  # Higher rank = more capacity
LORA_ALPHA = 32  # Typically 2*r
LORA_DROPOUT = 0.05  # Lower dropout for more learning

# ============================================================================
# LOAD MODEL & TOKENIZER
# ============================================================================

print("🚀 Loading Qwen2.5-1.5B-Instruct with Unsloth...")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,  # Auto-detect
    load_in_4bit=LOAD_IN_4BIT,
)

print("✅ Model loaded successfully!")

# ============================================================================
# CONFIGURE LORA
# ============================================================================

print("\n⚙️  Configuring LoRA adapters...")

model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    use_gradient_checkpointing="unsloth",  # Unsloth optimization
    random_state=42,
    use_rslora=False,
    loftq_config=None,
)

print("✅ LoRA configured!")
print(f"   - Rank: {LORA_R}")
print(f"   - Alpha: {LORA_ALPHA}")
print(f"   - Dropout: {LORA_DROPOUT}")

# ============================================================================
# LOAD DATASET
# ============================================================================

print("\n📦 Loading dataset...")

train_dataset = load_dataset('json', data_files=TRAIN_DATA, split='train')
val_dataset = load_dataset('json', data_files=VAL_DATA, split='train')

print(f"✅ Training samples: {len(train_dataset)}")
print(f"✅ Validation samples: {len(val_dataset)}")

# ============================================================================
# FORMATTING FUNCTION
# ============================================================================

def formatting_prompts_func(examples):
    """Format conversations using Qwen2.5 chat template"""
    # Handle both single example and batched examples
    if isinstance(examples["messages"][0], dict):
        # Single example: examples["messages"] is a list of message dicts
        messages = examples["messages"]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        return text
    else:
        # Batched examples: examples["messages"] is a list of conversations
        texts = []
        for messages in examples["messages"]:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            texts.append(text)
        return texts

# ============================================================================
# TRAINING ARGUMENTS
# ============================================================================

print("\n⚙️  Configuring training arguments...")

training_args = TrainingArguments(
    # Output
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    
    # Training
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    
    # Optimization
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine",
    warmup_ratio=WARMUP_RATIO,
    weight_decay=WEIGHT_DECAY,
    optim="adamw_8bit",  # Memory-efficient optimizer
    
    # Mixed precision
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    
    # Logging
    logging_steps=5,  # Log frequently for small dataset
    logging_dir=f"{OUTPUT_DIR}/logs",
    
    # Evaluation
    eval_strategy="steps",
    eval_steps=20,  # Evaluate every 20 steps
    
    # Saving
    save_strategy="steps",
    save_steps=20,
    save_total_limit=3,  # Keep best 3 checkpoints
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    
    # Other
    seed=42,
    report_to="none",  # Change to "wandb" if you want W&B logging
)

print("✅ Training configuration:")
print(f"   - Epochs: {EPOCHS}")
print(f"   - Batch size: {BATCH_SIZE} (effective: {BATCH_SIZE * GRAD_ACCUM})")
print(f"   - Learning rate: {LEARNING_RATE}")
print(f"   - Warmup ratio: {WARMUP_RATIO}")

# ============================================================================
# TRAINER
# ============================================================================

print("\n🏋️  Initializing trainer...")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    max_seq_length=MAX_SEQ_LENGTH,
    formatting_func=formatting_prompts_func,
    args=training_args,
    packing=False,  # Don't pack sequences for chat format
)

print("✅ Trainer initialized!")

# ============================================================================
# TRAIN
# ============================================================================

print("\n" + "="*60)
print("🚀 STARTING TRAINING")
print("="*60 + "\n")

trainer.train()

print("\n" + "="*60)
print("✅ TRAINING COMPLETE!")
print("="*60 + "\n")

# ============================================================================
# SAVE MODEL
# ============================================================================

print("💾 Saving final model...")

# Save LoRA adapters
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"✅ Model saved to: {OUTPUT_DIR}")

# Optional: Save merged model (base + LoRA)
print("\n💾 Saving merged model (optional)...")
model.save_pretrained_merged(
    f"{OUTPUT_DIR}_merged",
    tokenizer,
    save_method="merged_16bit",  # or "merged_4bit" for smaller size
)

print(f"✅ Merged model saved to: {OUTPUT_DIR}_merged")

print("\n🎉 All done! Your EmoGPT model is ready for testing!")
