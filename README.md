# DeepSeek-Coder 6.7B × NVIDIA OpenCodeInstruct — LoRA / QLoRA

A reproducible single-GPU supervised fine-tuning starter using **Transformers, Datasets, PEFT, TRL, and PyTorch**.

- **Model:** [deepseek-ai/deepseek-coder-6.7b-instruct](https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-instruct)
- **Dataset:** [nvidia/OpenCodeInstruct](https://huggingface.co/datasets/nvidia/OpenCodeInstruct)
- **Hardware:** T4 16GB or AWS G4dn (4-bit QLoRA), A100 40/80GB (BF16 LoRA)
- **Method:** Rank 16, alpha 32, dropout 0.05, q/k/v/o and feed-forward projection adapters.
- **Status:** Source and CPU preprocessing tests available; a full CUDA training run has **not** been completed or benchmarked.

## 🚀 One-click Google Colab T4 notebook

[**Open DeepSeek-Coder 6.7B QLoRA in Google Colab**](https://colab.research.google.com/github/Birajkakadiya/deepseek-finetuning/blob/main/colab/DeepSeek_Coder_6_7B_T4_QLoRA.ipynb)

1. In Colab choose **Runtime → Change runtime type → T4 GPU → Save** (subject to Colab availability and quotas).
2. Click **Runtime → Run all**. The default smoke test runs **2 optimizer steps**, **48 training samples**, **8 validation samples**, **512 tokens** with 4-bit NF4 QLoRA.
3. Inspect validation metrics and loss plot, then authorize the optional Google Drive cell to archive the LoRA adapter.
4. Only after the smoke test works, enable `RUN_LONG_TRAINING = True` in the optional 60-step cell and run that cell on its own.

The notebook downloads the code from this repository; **GitHub does not itself provide a free T4 GPU**. No long run is started automatically, and nothing is pushed to the repository from Colab.

## Quick start

On Ubuntu with a working CUDA-enabled NVIDIA GPU:

```bash
nvidia-smi
python3.11 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install 'torch>=2.6,<3'
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python train.py --preset t4 --max-steps 2 --max-train-samples 50 --max-eval-samples 10
```

For a longer T4/G4dn run:

```bash
python train.py --preset t4 --max-steps 60
python generate.py --adapter-dir outputs/t4/adapter --load-in-4bit --prompt "Write Python quicksort with tests."
```

For A100:

```bash
python train.py --preset a100 --max-steps 100
python generate.py --adapter-dir outputs/a100/adapter --prompt "Write Python merge sort with tests."
```

Training is **not** executed on ordinary CPU-only GitHub Actions runners. The manual `train_gpu.yml` workflow requires a configured self-hosted CUDA runner (or a provisioned paid GPU runner) with a label such as `gpu-t4` or `gpu-a100`. A100 GPU hardware is not supplied by this repository. If using AWS G4, choose **g4dn** (NVIDIA T4), not G4ad.

## Outputs

Adapters, tokenizer, metrics, and trainer state are stored under `outputs/<preset>/`. The GitHub Actions training job uploads an artifact when it succeeds. Outputs are gitignored.

## Implementation

The NVIDIA dataset is sampled via shuffled streaming, quality filtered optionally using `average_test_score`, then split into nonoverlapping train/eval rows. Training examples use conversational prompt/completion format, with **completion-only loss**. The base model stays frozen; PEFT trains LoRA modules. The T4 preset uses 4-bit NF4 quantization.

```text
train.py               Trainer and model loading
data_utils.py          Dataset validation, conversion, disjoint split
configs/t4.json        T4/G4dn defaults
configs/a100.json      A100 defaults
generate.py            Adapter inference
plot_loss.py           Loss-curve plotting
tests/test_data.py     CPU-only checks
.github/workflows/ci.yml          Data tests
.github/workflows/train_gpu.yml   Manual training on an existing GPU runner
```

## Caveats

- The NVIDIA dataset is large; these configs intentionally use only a small sample for an initial experiment. The initial run is not a claim of improved coding accuracy.
- Test code inside a generated model response must be sandboxed before execution.
- You are responsible for GPU machine costs and the terms of the model and dataset.
- Free GitHub-hosted `ubuntu-latest` jobs have no CUDA GPU; do not select them for training.
