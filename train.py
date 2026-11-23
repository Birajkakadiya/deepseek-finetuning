#!/usr/bin/env python3
"""Train DeepSeek-Coder 6.7B on NVIDIA OpenCodeInstruct with single-GPU LoRA/QLoRA."""
import argparse
import json
import sys
from pathlib import Path
from data_utils import record_to_conversation, select_disjoint_rows

BASE_MODEL = "deepseek-ai/deepseek-coder-6.7b-instruct"
DATASET = "nvidia/OpenCodeInstruct"
ROOT = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=["t4", "a100"], default="t4")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-eval-samples", type=int)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--min-test-score", type=float)
    parser.add_argument("--output-dir")
    parser.add_argument("--resume-from-checkpoint")
    args = parser.parse_args()
    cfg = json.loads((ROOT / "configs" / f"{args.preset}.json").read_text())
    for key in ("max_steps", "max_train_samples", "max_eval_samples", "max_length", "min_test_score", "output_dir"):
        v = getattr(args, key)
        if v is not None:
            cfg[key] = v
    if min(cfg["max_steps"], cfg["max_train_samples"], cfg["max_eval_samples"], cfg["max_length"]) <= 0:
        parser.error("max_steps, train/eval sample counts and max_length must all be positive")

    import torch
    from datasets import Dataset, load_dataset
    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        sys.exit("Exactly one NVIDIA CUDA GPU required; CPU-only GitHub runners cannot train.")
    gpu_name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if not cfg["load_in_4bit"] and vram < 32:
        sys.exit("Unquantized A100 preset requires >=32 GiB; choose --preset t4 for 4-bit QLoRA")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    set_seed(cfg["seed"])
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"GPU: {gpu_name} ({vram:.1f} GiB); loading streaming dataset", flush=True)
    dataset_stream = load_dataset(DATASET, split="train", streaming=True).shuffle(
        seed=cfg["seed"], buffer_size=cfg["shuffle_buffer"])
    training_rows, eval_rows = select_disjoint_rows(
        dataset_stream, cfg["max_train_samples"], cfg["max_eval_samples"], cfg["min_test_score"])
    training_data = Dataset.from_list([record_to_conversation(r) for r in training_rows])
    evaluation_data = Dataset.from_list([record_to_conversation(r) for r in eval_rows])

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    if tokenizer.chat_template is None:
        raise RuntimeError("Tokenizer does not provide a chat template")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    quantization = None
    if cfg["load_in_4bit"]:
        quantization = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
    kwargs = {"torch_dtype": dtype, "device_map": {"": 0}, "low_cpu_mem_usage": True}
    if quantization:
        kwargs["quantization_config"] = quantization
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, **kwargs)
    model.config.use_cache = False
    if cfg["load_in_4bit"]:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora = LoraConfig(
        r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"], bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    sf = SFTConfig(
        output_dir=str(output_dir),
        max_steps=cfg["max_steps"],
        max_length=cfg["max_length"],
        per_device_train_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        per_device_eval_batch_size=1,
        learning_rate=cfg["learning_rate"], lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        optim="paged_adamw_8bit" if cfg["load_in_4bit"] else "adamw_torch",
        bf16=dtype == torch.bfloat16, fp16=dtype == torch.float16,
        logging_steps=max(1, min(cfg["logging_steps"], cfg["max_steps"])),
        eval_strategy="steps", eval_steps=max(1, min(cfg["eval_steps"], cfg["max_steps"])),
        save_strategy="steps", save_steps=max(1, min(cfg["save_steps"], cfg["max_steps"])),
        save_total_limit=2, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        completion_only_loss=True, packing=False, report_to="none",
        seed=cfg["seed"], data_seed=cfg["seed"], save_safetensors=True)
    trainer = SFTTrainer(model=model, args=sf, train_dataset=training_data,
                         eval_dataset=evaluation_data, peft_config=lora, processing_class=tokenizer)
    trainer.model.print_trainable_parameters()
    result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    metrics = trainer.evaluate()
    adapter_dir = output_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    trainer.state.save_to_json(str(output_dir / "trainer_state.json"))
    metadata = {"base_model": BASE_MODEL, "dataset": DATASET, "gpu": gpu_name,
                "vram_gib": round(vram, 1), "preset": args.preset, "config": cfg,
                "train_rows": len(training_data), "eval_rows": len(evaluation_data),
                "train_metrics": result.metrics, "eval_metrics": metrics}
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    print(f"Adapter saved to {adapter_dir}; validation: {metrics}", flush=True)

if __name__ == "__main__":
    main()
