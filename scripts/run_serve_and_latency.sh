#!/bin/bash
# Script to start Ray Serve instances for each model and estimate inference latency.

set -e

echo "Starting Ray Serve for all model variants..."

# Serve original pretrained model
python ray_serve_final.py --quantization none --model_type pretrain

# Serve quantized pretrained models
python ray_serve_final.py --quantization 4bit --model_type pretrain
python ray_serve_final.py --quantization 8bit --model_type pretrain

# Serve quantized + fine-tuned models
python ray_serve_final.py --quantization 4bit --model_type finetuned
python ray_serve_final.py --quantization 8bit --model_type finetuned

echo "Serving complete. Estimating latency..."

# Latency estimation, average latency over 10 runs + memory usage
python estimate_latency.py --output_dir results_inference/original_pretrain --num_output 10
python estimate_latency.py --output_dir results_inference/4bit_pretrain --num_output 10
python estimate_latency.py --output_dir results_inference/8bit_pretrain --num_output 10
python estimate_latency.py --output_dir results_inference/4bit_finetuned --num_output 10
python estimate_latency.py --output_dir results_inference/8bit_finetuned --num_output 10

echo "Latency estimation complete."
