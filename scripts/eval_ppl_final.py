import os
import csv
import time
import torch
import logging
import numpy as np
from tqdm import tqdm 
from dataset import get_loaders
from argparse import ArgumentParser
from utils import get_model, count_params, set_seed
from transformers import BitsAndBytesConfig

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",

    level=logging.INFO
)
logger = logging.getLogger(__name__)

#  Optional: Check for flash attention
# try:
#     import flash_attn
#     attn_implementation = 'flash_attention_2'
#     logger.info(f'Using {attn_implementation}!')
# except ImportError:
#     attn_implementation = "eager"
#     logger.warning("flash_attention not installed, using eager attention")

# attn_implementation = "eager"

# Argument parser
def build_argparser():
    parser = ArgumentParser(description="Script to evaluate perplexity with a given model")
    parser.add_argument(
        "-m", "--model",
        default="meta-llama/Llama-3.1-8B",
        help="Model identifier (Hugging Face hub path or local path). Default: %(default)s"
    )
    parser.add_argument(
        "--quantization",
        choices=["2bit", "4bit", "8bit", "none"],
        default="none",
        help="Quantization type for model loading. Choices: [2bit, 4bit, 8bit, none]. Default: %(default)s"
    )

    parser.add_argument(
        "--model_type",
        choices=["finetuned", "pretrain"],
        default="pretrain",
        help="Model type for loading. Choices: [finetuned, pretrain]. Default: %(default)s"
    )

    parser.add_argument(
        "--attn_implementation",
        choices=['flash_attention_2', 'eager'],
        default="flash_attention_2",
        help="Attention implementation. Choices: [flash_attention_2, eager]. Default: %(default)s"
    )


    parser.add_argument(
        "--input_prompt", type=str, default="The Leaning Tower of Pisa is known for"
    )
    parser.add_argument("--num_output", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=float, default=50)
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument("--max_seq_len", type=int, default=128)

    parser.add_argument("--output_dir", type=str, default="results/llama-3.1-8b/ppl")

    parser.add_argument("--device", type=str, default="cuda", help="device")


    return parser


@torch.no_grad()
def llama_eval(model, test_lodaer, device):
    nlls = []
    for batch in tqdm(test_lodaer):
        batch = batch.to(device)
        output = model(batch)
        lm_logits = output.logits

        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = batch[:, 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(
            shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        nlls.append(loss)
    # print(torch.cat(nlls, dim=-1).mean())
    ppl = np.exp(torch.cat(nlls, dim=-1).mean().item())
    return ppl.item()


def eval_ppl(
    output_dir,
    model,
    tokenizer,
    datasets=["wikitext2", "ptb"],
    max_seq_len=128,
    batch_size=4,
    device="cuda",
    add_bos_to_every=False,
):
    filename = "ppl_bos.csv" if add_bos_to_every else "ppl.csv"
    csv_log_path = os.path.join(output_dir, filename)
    csv_header = []
    csv_value = []
    metric = {}
    for dataset in datasets:
        t0 = time.perf_counter()
        _, test_loader = get_loaders(
            dataset, tokenizer, max_seq_len, batch_size, add_bos_to_every
        )
        metric[dataset] = llama_eval(model, test_loader, device)

        print(
            f"PPL-{dataset}: {metric[dataset]} | add_bos_to_every: {add_bos_to_every} | time: {time.perf_counter()-t0:.1f}"
        )
        csv_header.append(f"ppl_{dataset}")
        csv_value.append(metric[dataset])

    mem = torch.cuda.memory_allocated() / 1024 / 1024
    print(f"Current GPU memory occupied: {mem} MiB")
    nparams = count_params(model)
    print(f"Params: {nparams}")

    with open(csv_log_path, "w") as logfile:
        logwriter = csv.writer(logfile, delimiter=",")
        logwriter.writerow(csv_header + ["params", "mem"])
        logwriter.writerow(csv_value + [nparams, mem])

def generate_txt(
    output_dir,
    model,
    tokenizer,
    input_prompt="The Leaning Tower of Pisa is known for",
    num_output=5,
    top_k=50,
    top_p=0.95,
    temperature=1.0,
    max_seq_len=128,
    device="cuda",
):
    # generate a few samples
    txt_path = os.path.join(output_dir, "gen_text.txt")
    inputs = tokenizer(input_prompt, return_tensors="pt")["input_ids"].to(device)
    input_len = inputs[0].size(0)

    with open(txt_path, "w", encoding="utf8") as f:
        f.write("=== input ===\n")
        f.write(f"{input_prompt}\n")

    for i in range(num_output):
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=inputs,
                do_sample=True,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                max_length=(input_len + max_seq_len),
                min_length=(
                    input_len + max_seq_len
                ),  # forced output length (to avoid <EOS> sampling)
                return_dict_in_generate=True,
            )
        s = generation_output.sequences[0]
        output_len = len(s)
        output = tokenizer.decode(s)

        print(f"=== output {i} | leng gen {output_len-input_len} + input {input_len}\n")
        print(output)

        with open(txt_path, "a", encoding="utf8") as f:
            f.write(
                f"=== output {i} | leng gen {output_len-input_len} + input {input_len}\n"
            )
            f.write(f"{output}\n")



if __name__ == "__main__":

    args = build_argparser().parse_args()

    model_name = args.model
    quant_type = args.quantization
    model_type = args.model_type
    output_dir = args.output_dir
    logger.info(f"Base Model selected: {model_name}")
    logger.info(f"Model type: {model_type}")
    logger.info(f"Quantization type: {quant_type}")

    attn_implementation = args.attn_implementation 
    if attn_implementation=='flash_attention_2':
        try:
            import flash_attn
            attn_implementation = 'flash_attention_2'
            logger.info(f'Using {attn_implementation}!')

        except ImportError:
            attn_implementation = "eager"
            logger.warning("flash_attention not installed, using eager attention")

    else:
        attn_implementation = "eager"
        logger.warning("flash_attention not installed, using eager attention")

    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    logger.info(f"Using compute dtype: {compute_dtype}")

    adapter_checkpoint = None #set default for adapter checkpoint

    quantization_config = None
    if quant_type == "4bit":
        logger.info("Applying 4-bit quantization.")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type="nf4"
        )

        if model_type == 'finetuned':
            adapter_checkpoint = "../Llama-3.1-8B-4bit_finetuned/checkpoint-5000/"
    elif quant_type == "8bit":
        # logger.info("Applying 8-bit quantization.")
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            # llm_int8_threshold=6.0
        )

        if model_type == 'finetuned':
            adapter_checkpoint = "../Llama-3.1-8B-8bit_finetuned/checkpoint-5600/"


    elif quant_type == "2bit":
        logger.info("Applying 8-bit quantization.")
        quantization_config = BitsAndBytesConfig(
            load_in_2bit=True
        )

    set_seed(args.seed)
    
    model, tokenizer, description = get_model(compute_dtype,
                                            attn_implementation, 
                                            quantization_config, 
                                            model_type=model_type,
                                            device=args.device,
                                            base_model=model_name,
                                            adapter_checkpoint= adapter_checkpoint)


    os.makedirs(args.output_dir, exist_ok=True)
    for add_bos_to_every in [True, False]:
        eval_ppl(output_dir,
                model,
                tokenizer,
                datasets=["wikitext2", "ptb"],
                max_seq_len=128,
                batch_size=4,
                device=args.device,
                add_bos_to_every=add_bos_to_every)

    generate_txt(
        output_dir=args.output_dir,
        model=model,
        tokenizer=tokenizer,
        input_prompt=args.input_prompt,
        num_output=args.num_output,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        max_seq_len=args.max_seq_len,
        device=args.device,
    )

                        






