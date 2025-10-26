# EmoGPT Fine-tuning Setup

Quick start guide for fine-tuning Qwen2.5-1.5B-Instruct on the EmoGPT dataset using Unsloth.

## Setup

### 1. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 2. Prepare Dataset

Convert the augmented JSONL to training format:

```powershell
python convert_dataset.py
```

This creates:
- `emogpt_train.jsonl` (~426 samples)
- `emogpt_val.jsonl` (~48 samples)

### 3. Train Model

```powershell
python train_emogpt.py
```

**Training specs:**
- Model: Qwen2.5-1.5B-Instruct (Unsloth optimized)
- Epochs: 5
- Batch size: 8 (effective 16 with gradient accumulation)
- Learning rate: 2e-4
- LoRA rank: 16
- Expected time: ~1-2 hours on A6000

**Outputs:**
- `./emogpt-qwen-1.5b/` - LoRA adapters only (~50-100MB)
- `./emogpt-qwen-1.5b_merged/` - Full merged model (~3GB)

### 4. Test Model

```powershell
python test_emogpt.py
```

## Configuration Details

### Hyperparameters (Slightly Aggressive)

```python
EPOCHS = 5                # More epochs for small dataset
BATCH_SIZE = 8            # Utilizing 48GB VRAM
LEARNING_RATE = 2e-4      # Higher LR for faster learning
LORA_R = 16               # Higher rank = more capacity
LORA_DROPOUT = 0.05       # Lower dropout for more learning
```

### Why Unsloth?

- **2x faster** training than standard implementations
- **60% less memory** usage with same performance
- Optimized kernels for QLoRA
- Seamless integration with HuggingFace ecosystem

## Monitoring Training

Watch for these metrics:

- **Training loss**: Should decrease to ~0.8-1.2
- **Validation loss**: Should track training loss closely
- **If val_loss >> train_loss**: Overfitting (reduce epochs or increase dropout)

## Expected Results

Given 474 samples:
- Token reduction: **15-25%** (baseline test)
- Training should converge smoothly
- Emoji usage should emerge by epoch 3-4

## Next Steps After Training

1. **Test on diverse prompts** - Check emoji appropriateness
2. **Expand dataset** - Add LMSYS conversations for 2K+ samples
3. **Re-train** - Better results with larger dataset
4. **Deploy** - Export to GGUF for production use

## Troubleshooting

### Out of Memory
Shouldn't happen with 48GB, but if it does:
- Reduce `BATCH_SIZE` to 4
- Reduce `MAX_SEQ_LENGTH` to 1024

### Slow Training
- Ensure CUDA is available: `torch.cuda.is_available()`
- Verify Unsloth is installed correctly
- Check GPU utilization: `nvidia-smi`

### Poor Results
- Check if dataset was properly filtered (augmented=true only)
- Try more epochs (7-8)
- Increase LoRA rank to 32

## Files Structure

```
Finetuning/
├── augmentedv1.jsonl          # Original dataset
├── convert_dataset.py          # Dataset preparation
├── train_emogpt.py            # Training script
├── test_emogpt.py             # Inference script
├── requirements.txt           # Dependencies
├── emogpt_train.jsonl         # Generated training data
├── emogpt_val.jsonl           # Generated validation data
└── emogpt-qwen-1.5b/          # Output model directory
```

## Notes

- This is a **baseline test** with 474 samples
- Results will improve significantly with 2K+ samples
- Current setup optimized for A6000 48GB VRAM
- Uses QLoRA (4-bit) for efficiency despite large VRAM
