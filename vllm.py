"""Benchmark offline inference throughput."""
import argparse
import json
import random
import time
from typing import List, Optional, Tuple

import torch
from vllm import LLM, SamplingParams
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          PreTrainedTokenizerBase)
from tqdm import tqdm

'''
python vllm.py --dataset scrambled_sampled_dataset.json --model llama2-70b-hf --tokenizer llama2-70b-hf --num-samples=10
'''

num_gpus = 8

def run_hf(
    requests: List[Tuple[str, int, int]],
    model: str,
    tokenizer: str,
    trust_remote_code: bool,
) -> float:

    # Initialize the VLLM model
    llm = LLM(model=model, tokenizer=tokenizer, dtype='float16', tensor_parallel_size=num_gpus, trust_remote_code=True, enforce_eager=True)
    # if llm.config.model_type == "llama":
    #     # To enable padding in the HF backend.
    #     tokenizer.pad_token = tokenizer.eos_token

    # Move the model to GPUs
    llm = torch.nn.DataParallel(llm, device_ids=list(range(num_gpus)))
    

    input_num_tokens = []
    output_num_tokens = []
    prompts = []
    # max_outlen = []
    params = []
    data_to_save = []
    start = time.perf_counter()

    for i in tqdm(range(len(requests))):
        prompt, prompt_len, output_len = requests[i]
        prompts.append(prompt)
        # max_outlen.append(output_len)
        sampling_params = SamplingParams(best_of=1, temperature=1, top_p=1, ignore_eos=True, max_tokens=output_len)
        params.append(sampling_params)
    
    # outputs = llm.module.generate(prompts, sampling_params=sampling_params, max_outlen=max_outlen)
    outputs = llm.module.generate(prompts, sampling_params=params)
    # outputs = llm.module.generate(prompt_token_ids=prompts_id, sampling_params=params)
    # Print the outputs
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        input_ids = output.prompt_token_ids
        output_ids = output.outputs[0].token_ids
        # print(f'input:{input_ids}')
        # print(f"input_token:{len(input_ids)}")
        # print(f'output:{output_ids}')
        # print(f"output_token:{len(output_ids)}")
        # print(f"Generated text: {generated_text}\n")
        input_num_tokens.append(len(input_ids))
        output_num_tokens.append(len(input_ids)+len(output_ids))
        data_to_save.append((generated_text, len(output_ids)))
        # print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

    end = time.perf_counter()
    # with open('log_vllm_no_kv.txt', "w") as file:
    #     for item in data_to_save:
    #         file.write(f"{item[0]},{item[1]}\n")



    return end - start, input_num_tokens, output_num_tokens



def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)

    if args.dataset is None:
        # Synthesize a prompt with the given input length.
        prompt = "hi" * (args.input_len - 1)
        requests = [(prompt, args.input_len, args.output_len)
                    for _ in range(args.num_samples)]

    else:
        with open(args.dataset) as f:
            requests = json.load(f)

    if args.num_samples is not None:
        requests = requests[0:args.num_samples]

    elapsed_time, input_num_tokens, output_num_tokens = run_hf(requests, args.model, args.tokenizer, args.trust_remote_code)
    prompt_num_tokens = sum(prompt_len for prompt_len in input_num_tokens)
    total_num_tokens = sum(output_len for output_len in output_num_tokens)
    print(f"Throughput: {len(requests) / elapsed_time:.2f} requests/s \n"
          f"Tokens/s: {total_num_tokens / elapsed_time:.2f} tokens/s \n"
          f"Prompt_num_tokens:{prompt_num_tokens:.2f} tokens \n"
          f"Total_num_tokens:{total_num_tokens:.2f} tokens \n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark the throughput.")
    parser.add_argument("--dataset", type=str, default=None, help="Path to the dataset.")
    parser.add_argument("--model", type=str, default="meta/llama2-70b")
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--input-len", type=int, default=None, help="Input prompt length for each request")
    parser.add_argument("--output-len", type=int, default=None, help="Output length for each request")
    parser.add_argument("--num-samples", type=int, default=None, help="Number of first few samples used for inference test")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument('--trust-remote-code',
                        action='store_true',
                        help='trust remote code from huggingface')
    parser.add_argument(
        '--dtype',
        type=str,
        default='auto',
        choices=['auto', 'half', 'float16', 'bfloat16', 'float', 'float32'],
        help='data type for model weights and activations. '
        'The "auto" option will use FP16 precision '
        'for FP32 and FP16 models, and BF16 precision '
        'for BF16 models.')
    args = parser.parse_args()
    if args.tokenizer is None:
        args.tokenizer = args.model
    if args.dataset is None:
        assert args.input_len is not None
        assert args.output_len is not None
    else:
        assert args.input_len is None
        assert args.output_len is None

    main(args)
