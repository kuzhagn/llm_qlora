from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import ray
from ray import serve
from fastapi import Request
import time

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from peft import PeftModel, PeftConfig
from utils import get_model, set_seed, count_params
from argparse import ArgumentParser


# Initialize Ray
ray.init()
serve.start(detached=True)

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



@serve.deployment(num_replicas=2, ray_actor_options={"num_gpus": 1})
class LLMDeployment:
    def __init__(self, compute_dtype, attn_implementation, 
                quantization_config, model_type, device, 
                 model_name, adapter_checkpoint):
        print("Loading model...")
      
        

        self.model, self.tokenizer, _ = get_model(compute_dtype,
                                                attn_implementation, 
                                                quantization_config, 
                                                model_type=model_type,
                                                device=device,
                                                base_model=model_name,
                                                adapter_checkpoint=adapter_checkpoint)

        self.model.eval() 

        # self.tokenizer = tokenizer
        print("Model loaded.")


    async def __call__(self, request: Request):
        payload = await request.json()
        prompt = payload.get("prompt", "")
        if not prompt:
            return {"error": "Missing prompt"}

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        start_time = time.time()
        output_ids = self.model.generate(input_ids=inputs["input_ids"], 
                                        do_sample=True,
                                        top_k=payload.get("top_k", 50),
                                        top_p=payload.get("top_p", 0.95),
                                        temperature=payload.get("temperature", 1.0),
                                        max_new_tokens=payload.get("max_new_tokens", 128),
)

        latency = time.time() - start_time

        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        mem_alloc = torch.cuda.memory_allocated() / 1024**2
        mem_reserved = torch.cuda.memory_reserved() / 1024**2

        # Measure memory used during generation
        print(f"[Generate] Allocated memory: {mem_alloc:.2f} MB")
        print(f"[Generate] Reserved memory:  {mem_reserved:.2f} MB")


        return {
            "output": output_text,
            "latency_sec": latency, 
            "memory_alloc": mem_alloc, 
            "memory_reserved":  mem_reserved
        }


if __name__ == "__main__":

    args = build_argparser().parse_args()

    model_name = args.model
    quant_type = args.quantization
    model_type = args.model_type
    output_dir = args.output_dir
    # logger.info(f"Base Model selected: {model_name}")
    # logger.info(f"Model type: {model_type}")
    # logger.info(f"Quantization type: {quant_type}")

    attn_implementation = args.attn_implementation 
    if attn_implementation=='flash_attention_2':
        try:
            import flash_attn
            attn_implementation = 'flash_attention_2'
            # logger.info(f'Using {attn_implementation}!')

        except ImportError:
            attn_implementation = "eager"
            # logger.warning("flash_attention not installed, using eager attention")

    else:
        attn_implementation = "eager"
        # logger.warning("flash_attention not installed, using eager attention")

    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    # logger.info(f"Using compute dtype: {compute_dtype}")

    adapter_checkpoint = None #set default for adapter checkpoint

    quantization_config = None
    if quant_type == "4bit":
        # logger.info("Applying 4-bit quantization.")
        quantization_config = BitsAndBytesConfig(load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type="nf4")

        if model_type == 'finetuned':
            adapter_checkpoint = "../Llama-3.1-8B-4bit_finetuned/checkpoint-5000/"

    elif quant_type == "8bit":
        # logger.info("Applying 8-bit quantization.")
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        if model_type == 'finetuned':
            adapter_checkpoint = "../Llama-3.1-8B-8bit_finetuned/checkpoint-5600/"


    elif quant_type == "2bit":
        # logger.info("Applying 8-bit quantization.")
        quantization_config = BitsAndBytesConfig(load_in_2bit=True)

    set_seed(args.seed)
    
    # Optional: bind to a FastAPI route (if using HTTP endpoints)
    serve.run(LLMDeployment.bind(
        compute_dtype=compute_dtype,
        attn_implementation=attn_implementation,
        quantization_config=quantization_config,
        model_type=model_type,
        device=args.device,
        model_name=model_name,
        adapter_checkpoint=adapter_checkpoint),
        route_prefix="/generate")

