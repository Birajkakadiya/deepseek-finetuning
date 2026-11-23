#!/usr/bin/env python3
"""Generate code with a trained PEFT LoRA adapter."""
import argparse
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter-dir", required=True)
    p.add_argument("--prompt", default="Write a Python prime number generator with tests.")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--load-in-4bit", action="store_true")
    a = p.parse_args()
    import torch
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    if not Path(a.adapter_dir).is_dir():
        raise FileNotFoundError(a.adapter_dir)
    config = PeftConfig.from_pretrained(a.adapter_dir)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    kwargs = {"torch_dtype": dtype, "device_map": {"": 0}}
    if a.load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
    base = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, **kwargs)
    model = PeftModel.from_pretrained(base, a.adapter_dir).eval()
    tokenizer = AutoTokenizer.from_pretrained(a.adapter_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": a.prompt}],
        add_generation_prompt=True, return_tensors="pt").to("cuda")
    with torch.no_grad():
        output = model.generate(inputs, max_new_tokens=a.max_new_tokens,
                                do_sample=False, eos_token_id=tokenizer.eos_token_id,
                                pad_token_id=tokenizer.pad_token_id)
    print(tokenizer.decode(output[0][inputs.shape[-1]:], skip_special_tokens=True))

if __name__ == "__main__":
    main()
