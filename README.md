# 🧠 LLM Compression Evaluation: Quantization and LoRA Fine-Tuning

This repository evaluates the impact of **quantization** (4-bit and 8-bit) and **LoRA fine-tuning** on large language models (LLMs). The goal is to understand trade-offs in:

- ✅ **Model performance** via perplexity (PPL)
- 🚀 **Inference latency** using Ray Serve
- 💾 **GPU memory usage**

---

## 🔬 Experimental Setup

**Base Model Used:**  
🧠 All experiments are conducted using the **LLaMA 3.1–8B** model variant (full-precision) as the starting point.

**Hardware:**  
🖥️ GPU: **NVIDIA A100 PCIe — 80GB HBM2e**  
Used for both training and inference.

| Spec                       | Value                        |
|---------------------------|------------------------------|
| GPU Memory                | 80GB HBM2e                   |
| TF32 Performance          | 156 TFLOPS                   |
| BF16 / FP16 Tensor Cores  | 312 TFLOPS                   |
| INT8 Tensor Cores         | 624 TOPS                     |
| Memory Bandwidth          | 1,935 GB/s                   |
| TDP                       | 300W                         |

---

## 🧪 Model Variants Assessed (Total: 5)

1. **Baseline model:**
   - `LLaMA 3.1–8B` (full-precision, no quantization or fine-tuning)

2. **Quantized Models:**
   - 4-bit quantized
   - 8-bit quantized

3. **Quantized + LoRA Fine-Tuned Models:**
   - 4-bit + LoRA fine-tuned
   - 8-bit + LoRA fine-tuned

---

## 📂 Project Structure

```
.
├── ppl_results/         # Stores perplexity evaluation outputs  
├── results_inference/   # Stores latency and inference results  
├── scripts/             # Contains Python abd Bash scripts  
├── requirements.txt     # Python dependencies  
└── README.md            # Project documentation (this file)
```



---

## ⚙️ Setup Instructions

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
```
accelerate config
```


📜 Bash Scripts

🔧 run_finetune_and_ppl.sh
Runs QLoRA fine-tuning on 4-bit and 8-bit models on alpaca-cleaned dataset
Evaluates perplexity for all variants on wikitext2 and ptb testsets
```
bash run_finetune_and_ppl.sh
```

🚀 run_serve_and_latency.sh
Launches Ray Serve for each model
Estimates latency and memory usage over multiple outputs

```
bash run_serve_and_latency.sh
```

🧪 Evaluation Outputs

Perplexity results: ppl_results/
Latency results: results_inference/
Each subfolder is named based on quantization level and whether LoRA fine-tuning was applied.

🙌 Acknowledgements

Hugging Face Transformers
Ray Serve

