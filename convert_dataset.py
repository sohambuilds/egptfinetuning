"""
Convert augmentedv1.jsonl to training format for fine-tuning.
Splits into train/val (90/10) and filters for augmented samples only.
"""

import json
from datasets import Dataset

def convert_to_chat_format(input_file, output_prefix):
    """Convert JSONL to HuggingFace chat format"""
    data = []
    skipped = 0
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            conv = json.loads(line)
            
            # Only use successfully augmented conversations
            if conv['metadata'].get('augmented') == True:
                data.append({
                    'messages': conv['messages']
                })
            else:
                skipped += 1
    
    print(f"✅ Loaded {len(data)} augmented conversations")
    print(f"⏭️  Skipped {skipped} non-augmented conversations")
    
    # Create HuggingFace dataset
    dataset = Dataset.from_list(data)
    
    # Split into train/validation (90/10)
    split = dataset.train_test_split(test_size=0.1, seed=42)
    
    # Save as JSONL
    split['train'].to_json(f"{output_prefix}_train.jsonl")
    split['test'].to_json(f"{output_prefix}_val.jsonl")
    
    print(f"\n📊 Training samples: {len(split['train'])}")
    print(f"📊 Validation samples: {len(split['test'])}")
    print(f"\n💾 Saved to:")
    print(f"   - {output_prefix}_train.jsonl")
    print(f"   - {output_prefix}_val.jsonl")
    
    return split

if __name__ == "__main__":
    convert_to_chat_format(
        'augmentedv1.jsonl',
        'emogpt'
    )
