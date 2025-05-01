import os
import logging
from datetime import datetime
from argparse import ArgumentParser
import pandas as pd
import matplotlib.pyplot as plt

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    pipeline, 
    DataCollatorForSeq2Seq
)
from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    TaskType,
    get_peft_model
)

from LLMPruner import Prompter, ZeroPrompter


def build_argparser():
    parser = ArgumentParser()
    parser.add_argument("--quantization", type=str, choices=["4bit", "8bit", "none"], default="4bit")

    parser.add_argument("--lora_r", type=int, default=8, help="lora r")
    parser.add_argument("--lora_alpha", type=int, default=16, help="lora alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="lora dropout")
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,down_proj,up_proj",
        help="lora target modules",
    )

    parser.add_argument("--num_epochs", type=int, default=5, help="number of epochs")
    parser.add_argument(
        "--learning_rate", type=float, default=3e-4, help="learning rate"
    )
    parser.add_argument("--cutoff_len", type=int, default=256, help="cutoff length")
    parser.add_argument(
        "--val_set_size", type=int, default=2000, help="validation set size"
    )
    parser.add_argument(
        "--prompt_template_name",
        type=str,
        default="alpaca",
        help="The prompt template to use, will default to alpaca.",
    )
    parser.add_argument(
        "--no_instruction",
        action="store_true",
        default=False,
        help="Whether to use the instruction template or not.",
    )

    return parser

args = build_argparser().parse_args()

# Base model name
model_name = "meta-llama/Llama-3.1-8B"
prompter = Prompter('alpaca')
cutoff_len = 256
train_on_inputs = False
add_eos_token = False
data_path = 'yahma/alpaca-cleaned'
val_set_size = 2000

target_modules = ['k_proj', 'q_proj', 'v_proj', 'o_proj', "gate_proj", "down_proj", "up_proj"]
model_id = model_name.split("/")[-1]  # -> "Llama-3.1-8B"
quant_type = args.quantization.lower()  # normalize just in case

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
tag = f"{model_id}-{quant_type}"

logger_file = f"logs/{tag}_{timestamp}.log"
excel_file = f"logs/{tag}_losses.xlsx"
save_path = f"saved/{tag}_model"

# Make sure folders exist
os.makedirs("logs", exist_ok=True)
os.makedirs("saved", exist_ok=True)

# ------------------ Logging Setup ------------------
logging.basicConfig(
    filename=logger_file,
    filemode='w',
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')
console.setFormatter(formatter)
logging.getLogger().addHandler(console)
print = lambda *a, **kw: logging.info(" ".join(map(str, a)))

# ------------------ Flash Attention ------------------
try:
    import flash_attn
    attn_implementation = 'flash_attention_2'
    print(f"Using {attn_implementation}")
except ImportError:
    attn_implementation = None
    print("flash_attention not installed")


def print_trainable_parameters(model):
    trainable_params, all_param = 0, 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}")


class LossLoggerCallback(TrainerCallback):
    def __init__(self):
        self.train_losses = []
        self.eval_losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            step = state.global_step
            if 'loss' in logs:
                self.train_losses.append((step, logs['loss']))
            if 'eval_loss' in logs:
                self.eval_losses.append((step, logs['eval_loss']))

            # Log to logger file (this ensures every metric is logged)
            logging.info(f"[Step {step}] Metrics: {logs}")


def tokenize(prompt, add_eos_token=True):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_len,
        padding=False,
        return_tensors=None,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < cutoff_len
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)
    result["labels"] = result["input_ids"].copy()
    return result

def generate_and_tokenize_prompt(data_point):
    full_prompt = prompter.generate_prompt(
        data_point["instruction"],
        data_point["input"],
        data_point["output"],
    )
    tokenized_full_prompt = tokenize(full_prompt)
    
    if not train_on_inputs:
        user_prompt = prompter.generate_prompt(data_point["instruction"], data_point["input"])
        tokenized_user_prompt = tokenize(user_prompt, add_eos_token=add_eos_token)
        user_prompt_len = len(tokenized_user_prompt["input_ids"])
        
        if add_eos_token:
            user_prompt_len -= 1
            
        tokenized_full_prompt["labels"] = [
            -100
        ] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]
    
    return tokenized_full_prompt



def split_and_tokenizer(test_data, tokenizer, seq_len, field_name):
    test_ids = tokenizer("\n\n".join(test_data[field_name]), return_tensors="pt").input_ids[0]
    nsamples = test_ids.numel() // seq_len
    test_set = []
    for i in range(nsamples):
        batch = test_ids[(i * seq_len):((i + 1) * seq_len)]
        test_set.append({"input_ids": batch, "labels": batch})
    return test_set




# ------------------ Configuration ------------------
model_name = "meta-llama/Llama-3.1-8B"
# lora_r, lora_alpha, lora_dropout = 16, 16, 0.05
# target_modules = ['k_proj', 'q_proj', 'v_proj', 'o_proj', "gate_proj", "down_proj", "up_proj"]
# batch_size = 4

# ------------------ Dataset ------------------
print("Loading dataset...")
dataset = load_dataset(data_path)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, add_eos_token=True, use_fast=True)

# tokenizer.pad_token = tokenizer.eos_token
if tokenizer.pad_token is None:
    if tokenizer.unk_token is not None:
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

print("Loading dataset...")


# Load Train Dataset
data = load_dataset(data_path)
train_val = data["train"].train_test_split(
    test_size=args.val_set_size, shuffle=True, seed=42
)
print("Tokenizing dataset...")

train_data = train_val["train"].shuffle().map(generate_and_tokenize_prompt)

val_data = {
    data_path: train_val["test"].shuffle().map(generate_and_tokenize_prompt),
}

# ------------------ Model ------------------
compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

quant_config = None
if args.quantization == "4bit":
    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_storage=compute_dtype)
elif args.quantization == "8bit":
    quant_config = BitsAndBytesConfig(load_in_8bit=True)
elif args.quantization == "none":
    quant_config = None

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=compute_dtype,
    trust_remote_code=True,
    device_map="auto",
    quantization_config=quant_config,
    attn_implementation=attn_implementation,
)


model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
    r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
    task_type=TaskType.CAUSAL_LM, target_modules=target_modules
)
model = get_peft_model(model, peft_config)
model = model.bfloat16()

print_trainable_parameters(model)

# ------------------ Training ------------------
loss_logger = LossLoggerCallback()
training_args = TrainingArguments(
    output_dir = f"./{tag}_finetuned",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    #  warmup_steps=100,
    num_train_epochs=2,
    logging_steps=10,
    logging_first_step=True,
    optim="adamw_torch",
    eval_strategy="steps",
    save_strategy="steps",
    eval_steps=100,
    save_steps=200,
    learning_rate= 1e-4 ,
    # logging_dir="./logs",
    # logging_steps=10,
    save_total_limit=2,
    # eval_strategy="steps",
    # eval_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="{}_loss".format(data_path),
    # report_to="wandb"
        )

       

trainer = Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    processing_class=tokenizer,
    callbacks=[loss_logger],
    args=training_args,
    data_collator=DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
))

model.config.use_cache = False
print("Starting training...")
trainer.train()

# ------------------ Save Losses ------------------
train_steps, train_values = zip(*loss_logger.train_losses)
eval_dict = dict(loss_logger.eval_losses)
eval_values = [eval_dict.get(step, None) for step in train_steps]

df = pd.DataFrame({
    "step": train_steps,
    "loss": train_values,
    "eval_loss": eval_values
})
df.to_excel(excel_file, index=False)
print(f"Saved losses to {excel_file}")

plt.figure(figsize=(10, 6))
plt.plot(train_steps, train_values, label="Train Loss")
plt.plot(train_steps, eval_values, label="Eval Loss", linestyle="--")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Training and Evaluation Losses")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(excel_file.replace(".xlsx", ".png"))
plt.show()

# ------------------ Save Final Model ------------------
print("Merging LoRA weights and saving final model...")
merged_model = model.merge_and_unload()
merged_model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"Merged models saved to {save_path}")
print(f"Logs saved to {logger_file}")
print("Training complete.")