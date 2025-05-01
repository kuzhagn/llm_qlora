#!/bin/bash
# Script to run QLoRA fine-tuning and evaluate perplexity for original, quantized, and fine-tuned models.

set -e

echo "Starting QLoRA fine-tuning..."

# Fine-tune 4-bit quantized model on alpaca dataset
accelerate launch --mixed_precision bf16 qlora_accelerate_alpaca.py --quantization 4bit

# Fine-tune 8-bit quantized model alpaca dataset
accelerate launch --mixed_precision bf16 qlora_accelerate_alpaca.py --quantization 8bit

echo "Fine-tuning complete. Starting perplexity evaluation..."

# Perplexity for QLoRA fine-tuned models 
python eval_ppl_qlora.py

# Evaluate perplexity on original model (no quantization, no fine-tuning)
python eval_ppl_final.py --quantization none --model_type pretrain --output_dir results_original_pretrain

# Evaluate perplexity on quantized models (wikitext2 and ptb datasets)
python eval_ppl_final.py --quantization 4bit --model_type pretrain --output_dir results_4bit_pretrain
python eval_ppl_final.py --quantization 8bit --model_type pretrain --output_dir results_8bit_pretrain

# Evaluate perplexity on QLoRA fine-tuned models (wikitext2 and ptb datasets)
python eval_ppl_final.py --quantization 4bit --model_type finetuned --output_dir results_4bit_finetuned
python eval_ppl_final.py --quantization 8bit --model_type finetuned --output_dir results_8bit_finetuned

echo "Perplexity evaluation complete."
