import argparse
import json
import random
import time
from typing import List, Optional, Tuple

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,PreTrainedTokenizerBase,LogitsProcessorList,
    MinLengthLogitsProcessor,StoppingCriteriaList,MaxLengthCriteria,AutoConfig)
import os
from tqdm import tqdm

'''
python speculative.py --dataset scrambled_sampled_dataset.json --model llama2-70b-hf --num-samples=10
'''

def run_hf(
    requests: List[Tuple[str, int, int]],
    model: str,
    tokenizer: PreTrainedTokenizerBase,
    trust_remote_code: bool,
) -> float:
    
    assistant_model = AutoModelForCausalLM.from_pretrained("llama-2-7b-hf", device_map = 'auto')
    model = AutoModelForCausalLM.from_pretrained(model, device_map = "auto" , torch_dtype=torch.float16, trust_remote_code=trust_remote_code)
    # set pad_token_id to eos_token_id because GPT2 does not have a PAD token
    model.generation_config.pad_token_id = model.generation_config.eos_token_id
    model.eval()
    assistant_model.eval()

    input_num_tokens = []
    output_num_tokens = []
    data_to_save = []
    start = time.perf_counter()
    for i in tqdm(range(len(requests))):
        with torch.no_grad():
            prompt, prompt_len, output_len = requests[i]
            # Generate the sequences.
            input_ids = tokenizer(prompt, return_tensors="pt",
                                padding=True).input_ids.to(assistant_model.device)
            logits_processor = LogitsProcessorList(
                [
                    MinLengthLogitsProcessor(10, eos_token_id=model.generation_config.eos_token_id),
                ]
            )
            stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=prompt_len + output_len)])

            outputs = model.assisted_decoding(
                input_ids,
                assistant_model=assistant_model,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                do_sample=False,
                num_return_sequences=1,
                num_beams=1,
                temperature=1.0,
                top_p=1.0,
                use_cache=True
            )
            # Include the decoding time.
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # print(generated_text)
            # print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
            input_num_tokens.append(len(input_ids[0]))
            output_num_tokens.append(len(outputs[0]))
            data_to_save.append((generated_text, len(outputs[0])-len(input_ids[0])))

    end = time.perf_counter()
    with open('speculative.txt', "w") as file:
        for item in data_to_save:
            file.write(f"{item[0]},{item[1]}\n")
    return end - start, input_num_tokens, output_num_tokens




def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)

    # Sample the requests.
    tokenizer = AutoTokenizer.from_pretrained(
        "llama-2-7b-hf", trust_remote_code=args.trust_remote_code)
    tokenizer.pad_token = tokenizer.eos_token
    
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

    elapsed_time, input_num_tokens, output_num_tokens = run_hf(requests, args.model, tokenizer,  args.trust_remote_code)
    prompt_num_tokens = sum(prompt_len for prompt_len in input_num_tokens)
    total_num_tokens = sum(output_len for output_len in output_num_tokens)
    print(f"Throughput: {len(requests) / elapsed_time:.2f} requests/s \n"
          f"Tokens/s: {total_num_tokens / elapsed_time:.2f} tokens/s \n"
          f"Prompt_num_tokens:{prompt_num_tokens:.2f} tokens \n"
          f"Total_num_tokens:{total_num_tokens:.2f} tokens \n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark the throughput.")
    parser.add_argument("--dataset", type=str, default=None, help="Path to the dataset.")
    parser.add_argument("--model", type=str, default="llama2-70b")
    parser.add_argument("--tokenizer", type=str, default="llama2-70b")
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
