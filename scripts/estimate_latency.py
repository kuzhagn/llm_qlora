import os
import requests
import time
from argparse import ArgumentParser

def build_argparser():
    parser = ArgumentParser(description="Script to evaluate inference latency")

    parser.add_argument("--input_prompt", type=str, default="The Leaning Tower of Pisa is known for")
    parser.add_argument("--num_output", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--output_dir", type=str, default="results/latency_llama-3.1-8b/")
    parser.add_argument("--device", type=str, default="cuda", help="device")

    return parser


def generate_txt_from_serve(
    output_dir,
    input_prompt="The Leaning Tower of Pisa is known for",
    num_output=10,
    top_k=50,
    top_p=0.95,
    temperature=1.0,
    max_seq_len=128,
    endpoint_url="http://localhost:8000/generate",
):
    txt_path = os.path.join(output_dir, "gen_text.txt")
    latencies = []
    mem_allocs = []
    mem_reserved_vals = []

    os.makedirs(output_dir, exist_ok=True)

    with open(txt_path, "w", encoding="utf8") as f:
        f.write("=== input ===\n")
        f.write(f"{input_prompt}\n\n")

    for i in range(num_output):
        payload = {
            "prompt": input_prompt,
            "top_k": top_k,
            "top_p": top_p,
            "temperature": temperature,
            "max_new_tokens": max_seq_len,
        }

        start = time.time()
        response = requests.post(endpoint_url, json=payload)
        end = time.time()

        if response.status_code != 200:
            print(f"Request failed at sample {i}, status {response.status_code}")
            continue

        result = response.json()
        output = result["output"]
        latency = result.get("latency_sec", end - start)
        memory_alloc = result.get("memory_alloc", -1)
        memory_reserved = result.get("memory_reserved", -1)

        latencies.append(latency)
        mem_allocs.append(memory_alloc)
        mem_reserved_vals.append(memory_reserved)

        print(f"=== output {i}\n{output}\nLatency: {latency:.3f}s | Alloc: {memory_alloc:.2f} MB | Reserved: {memory_reserved:.2f} MB\n")

        with open(txt_path, "a", encoding="utf8") as f:
            f.write(f"=== output {i}\n{output}\n")
            f.write(f"Latency: {latency:.3f}s | Allocated: {memory_alloc:.2f} MB | Reserved: {memory_reserved:.2f} MB\n\n")

    if latencies:
        avg_latency = sum(latencies) / len(latencies)
        avg_alloc = sum(mem_allocs) / len(mem_allocs)
        avg_reserved = sum(mem_reserved_vals) / len(mem_reserved_vals)

        print(f"\nAverage Latency: {avg_latency:.3f} sec")
        print(f"Average Allocated Memory: {avg_alloc:.2f} MB")
        print(f"Average Reserved Memory: {avg_reserved:.2f} MB")

        with open(txt_path, "a", encoding="utf8") as f:
            f.write("=== Summary ===\n")
            f.write(f"Average Latency: {avg_latency:.3f} sec\n")
            f.write(f"Average Allocated Memory: {avg_alloc:.2f} MB\n")
            f.write(f"Average Reserved Memory: {avg_reserved:.2f} MB\n")

    return latencies


if __name__ == "__main__":
    args = build_argparser().parse_args()

    generate_txt_from_serve(
        output_dir=args.output_dir,
        input_prompt=args.input_prompt,
        num_output=args.num_output,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        max_seq_len=args.max_seq_len,
        endpoint_url="http://localhost:8000/generate"
    )
