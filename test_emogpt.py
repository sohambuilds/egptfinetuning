"""
Test inference with the fine-tuned EmoGPT model.
"""

import torch
from unsloth import FastLanguageModel

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_PATH = "./emogpt-qwen-1.5b"  # Path to your fine-tuned model
MAX_SEQ_LENGTH = 2048

# Test prompts
TEST_PROMPTS = [
    "How do I install Python on my computer?",
    "Can you explain what machine learning is in simple terms?",
    "What's the best way to learn programming as a beginner?",
    "I'm feeling overwhelmed with my coding project. Any advice?",
    "Explain the difference between a list and a tuple in Python.",
    "What are some good resources for learning web development?",
]

# ============================================================================
# LOAD MODEL
# ============================================================================

print("🤖 Loading EmoGPT model...")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=True,
)

# Enable fast inference mode
FastLanguageModel.for_inference(model)

print("✅ Model loaded!\n")
print("="*60)

# ============================================================================
# GENERATE RESPONSES
# ============================================================================

for i, prompt in enumerate(TEST_PROMPTS, 1):
    print(f"\n🧪 Test {i}/{len(TEST_PROMPTS)}")
    print(f"👤 User: {prompt}")
    
    # Format as chat
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda")
    
    # Generate
    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        use_cache=True,
    )
    
    # Decode
    response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
    
    print(f"🤖 EmoGPT: {response}")
    print("-"*60)

print("\n✅ Testing complete!")
