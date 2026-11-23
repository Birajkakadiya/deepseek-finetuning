#!/usr/bin/env python3
"""Publish a verified LoRA-only adapter ZIP to a NEW Hugging Face model repository.

The default is a safe preview. Publishing requires an explicit --publish flag.
Credentials come only from HF_TOKEN environment variable (never a CLI argument).
"""
import argparse
import json
import os
import shutil
import struct
import tempfile
from pathlib import Path
from zipfile import ZipFile, BadZipFile

BASE_MODEL = "deepseek-ai/deepseek-coder-6.7b-instruct"
DATASET = "nvidia/OpenCodeInstruct"
FILES = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
)
MAX_SAFETENSORS_HEADER_BYTES = 4_000_000


def _read_json(archive, name):
    return json.loads(archive.read(name).decode("utf-8"))


def inspect_archive(zip_path):
    """Read and validate archive metadata without loading model weights into memory."""
    if not zip_path.is_file():
        raise FileNotFoundError(f"Adapter ZIP not found: {zip_path}")
    try:
        with ZipFile(zip_path) as archive:
            names = set(archive.namelist())
            expected = {"adapter/" + item for item in FILES}
            expected |= {"run_metadata.json", "trainer_state.json"}
            missing = expected - names
            if missing:
                raise ValueError("Missing expected files: " + ", ".join(sorted(missing)))
            bad_file = archive.testzip()
            if bad_file:
                raise ValueError(f"Damaged ZIP member: {bad_file}")
            adapter = _read_json(archive, "adapter/adapter_config.json")
            metrics = _read_json(archive, "run_metadata.json")
            trainer = _read_json(archive, "trainer_state.json")
            if adapter.get("base_model_name_or_path") != BASE_MODEL:
                raise ValueError("Unexpected base model in adapter_config.json")
            if adapter.get("peft_type") != "LORA":
                raise ValueError("Archive does not contain a LoRA adapter")
            if metrics.get("base_model") != BASE_MODEL or metrics.get("dataset") != DATASET:
                raise ValueError("The run metadata does not match this project")
            if trainer.get("global_step") != metrics.get("config", {}).get("max_steps"):
                raise ValueError("Trainer global step does not match reported max_steps")
            with archive.open("adapter/adapter_model.safetensors") as weights:
                header_len = struct.unpack("<Q", weights.read(8))[0]
                if not (0 < header_len < MAX_SAFETENSORS_HEADER_BYTES):
                    raise ValueError("Invalid safetensors header size")
                header = json.loads(weights.read(header_len).decode("utf-8"))
                tensors = [v for k, v in header.items() if k != "__metadata__"]
                if not tensors:
                    raise ValueError("No tensors in adapter_model.safetensors")
            return adapter, metrics, trainer, len(tensors)
    except BadZipFile as exc:
        raise ValueError("Invalid ZIP archive") from exc


def make_model_card(repo_id, cfg, metadata, trainer, num_tensors):
    train = metadata["train_metrics"]
    evaluation = metadata["eval_metrics"]
    steps = trainer["global_step"]
    params = (
        "smoke test" if steps <= 10 else
        "experimental adapter"
    )
    # YAML metadata includes a reference to the actual upstream (custom) model license.
    return f"""---
language:
- en
license: other
license_name: deepseek
license_link: https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-instruct/blob/main/LICENSE
base_model: {BASE_MODEL}
datasets:
- {DATASET}
library_name: peft
pipeline_tag: text-generation
tags:
- lora
- qlora
- transformers
- trl
- code-generation
- experimental
---

# DeepSeek-Coder 6.7B × OpenCodeInstruct — T4 LoRA ({params})

This repository contains **LoRA adapter weights only**, not the full base model.
The adapter must be loaded on top of [{BASE_MODEL}](https://huggingface.co/{BASE_MODEL}).

> **Experimental training checkpoint:** {steps} optimizer steps on {metadata['train_rows']} training
> examples and {metadata['eval_rows']} validation examples. No independent code-generation
> benchmark or comparison with the base model has been performed. Do **not** interpret
> token accuracy as a pass rate on programming problems.

## Provenance and attribution

- Base model: [DeepSeek-Coder 6.7B Instruct](https://huggingface.co/{BASE_MODEL}).
- Training dataset: [NVIDIA OpenCodeInstruct](https://huggingface.co/datasets/{DATASET}),
  released by NVIDIA under **Creative Commons Attribution 4.0 (CC BY 4.0)**.
- Training scripts: [GitHub project](https://github.com/Birajkakadiya/deepseek-finetuning).
- Base-model use remains subject to the [DeepSeek Model License](https://huggingface.co/{BASE_MODEL}/blob/main/LICENSE).
  This model-card metadata does not relicense the base model or dataset.

## Training configuration

| Parameter | Value |
| --- | --- |
| Base model | `{BASE_MODEL}` |
| Dataset | `{DATASET}` |
| Hardware | {metadata['gpu']} ({metadata['vram_gib']} GiB VRAM) |
| Method | 4-bit NF4 QLoRA |
| LoRA rank / alpha / dropout | {cfg['r']} / {cfg['lora_alpha']} / {cfg['lora_dropout']} |
| Sequence length | {metadata['config']['max_length']} |
| Learning rate | {metadata['config']['learning_rate']} |
| Training steps | {steps} |
| Training examples | {metadata['train_rows']} |
| Validation examples | {metadata['eval_rows']} |
| Saved LoRA tensors | {num_tensors} |

## Reported small-sample metrics

| Metric | Value |
| --- | --- |
| Training loss | {train['train_loss']:.4f} |
| Validation loss | {evaluation['eval_loss']:.4f} |
| Validation mean token accuracy | {evaluation['eval_mean_token_accuracy']:.2%} |

Validation metrics are based on just {metadata['eval_rows']} examples and
**are not an evaluation of coding task correctness, generalization, or quality**.

## Loading the adapter on a T4 GPU

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

base_id = "{BASE_MODEL}"
adapter_id = "{repo_id}"
quant = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)
base = AutoModelForCausalLM.from_pretrained(
    base_id, device_map="auto", quantization_config=quant,
    torch_dtype=torch.float16,
)
model = PeftModel.from_pretrained(base, adapter_id).eval()
tokenizer = AutoTokenizer.from_pretrained(adapter_id)
prompt = [{{"role": "user", "content": "Write a Python binary search function with tests."}}]
input_ids = tokenizer.apply_chat_template(
    prompt, add_generation_prompt=True, return_tensors="pt"
).to("cuda")
with torch.no_grad():
    output = model.generate(input_ids, max_new_tokens=256, do_sample=False,
                            pad_token_id=tokenizer.eos_token_id)
print(tokenizer.decode(output[0][input_ids.shape[-1]:], skip_special_tokens=True))
```

## Risks and limitations

- This is a proof-of-pipeline release, **not** a validated performance improvement.
- Generated code may be incorrect, insecure, or incomplete; run tests in an isolated sandbox.
- Training data attribution: NVIDIA, [OpenCodeInstruct](https://huggingface.co/datasets/{DATASET}), CC BY 4.0.
- Full-base-model weights are not included. Use of the model is subject to the original license.
"""


def prepare_upload(zip_path, staging, repo_id):
    adapter, metrics, trainer, tensor_count = inspect_archive(zip_path)
    with ZipFile(zip_path) as z:
        for basename in FILES:
            with z.open("adapter/" + basename) as src, (staging / basename).open("wb") as dest:
                shutil.copyfileobj(src, dest)
        (staging / "training_metrics.json").write_text(
            json.dumps({"metadata": metrics, "trainer_state": trainer}, indent=2) + "\n",
            encoding="utf-8")
        if "loss.png" in z.namelist():
            with z.open("loss.png") as src, (staging / "loss.png").open("wb") as dest:
                shutil.copyfileobj(src, dest)
    (staging / "README.md").write_text(
        make_model_card(repo_id, adapter, metrics, trainer, tensor_count),
        encoding="utf-8")
    return metrics, tensor_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=Path, required=True, help="ZIP saved by the training notebook")
    parser.add_argument("--repo-name", default="deepseek-coder-6.7b-opencodeinstruct-lora-t4-smoke")
    parser.add_argument("--visibility", choices=("public", "private"), default="public")
    parser.add_argument("--publish", action="store_true",
                        help="Actually create a new Hugging Face model repo and upload it")
    args = parser.parse_args()
    name = args.repo_name
    if not name or "/" in name or name.startswith("."):
        parser.error("--repo-name must be one Hugging Face repository name, without a namespace")
    adapter, metadata, trainer, count = inspect_archive(args.zip)
    print(f"Archive verified: {count} LoRA tensors, {trainer['global_step']} optimizer steps, "
          f"{metadata['train_rows']} training and {metadata['eval_rows']} evaluation examples")
    print("Target repository name:", name)
    print("Visibility:", args.visibility)
    if not args.publish:
        print("DRY RUN ONLY: no remote repository was created and nothing was uploaded.")
        print("Add --publish only after checking the card and visibility.")
        return

    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("Set HF_TOKEN securely in Colab Secrets and the process environment.")
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    whoami = api.whoami(token=token)
    namespace = whoami.get("name")
    if not namespace:
        raise RuntimeError("Could not determine Hugging Face account username")
    repo_id = f"{namespace}/{name}"
    if api.repo_exists(repo_id=repo_id, repo_type="model", token=token):
        raise RuntimeError(f"Model repo {repo_id} already exists; choose another name to avoid overwriting")
    with tempfile.TemporaryDirectory(prefix="hf_lora_upload_") as temp:
        staging = Path(temp)
        prepare_upload(args.zip, staging, repo_id)
        print(f"Creating NEW {args.visibility} model repository: https://huggingface.co/{repo_id}")
        api.create_repo(
            repo_id=repo_id, repo_type="model", private=(args.visibility == "private"),
            exist_ok=False, token=token)
        api.upload_folder(
            folder_path=str(staging), repo_id=repo_id, repo_type="model",
            commit_message="Publish verified OpenCodeInstruct T4 LoRA smoke-test adapter",
            token=token)
        uploaded = api.list_repo_files(repo_id=repo_id, repo_type="model", token=token)
        required = {"adapter_config.json", "adapter_model.safetensors", "README.md"}
        if not required.issubset(set(uploaded)):
            raise RuntimeError("Upload returned but essential files are absent from Hub")
        print("PUBLISHED:", f"https://huggingface.co/{repo_id}")
        print("Confirmed files:", ", ".join(uploaded))


if __name__ == "__main__":
    main()
