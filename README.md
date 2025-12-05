# Noisy Coconut 🥥

**Testing Noise Robustness in Continuous Latent Reasoning**

[![arXiv](https://img.shields.io/badge/arXiv-2412.06769-b31b1b.svg)](https://arxiv.org/abs/2412.06769)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

**Noisy Coconut** extends the [Coconut (Continuous Chain-of-Thought)](https://arxiv.org/abs/2412.06769) framework to study how noise affects continuous latent reasoning in large language models. Instead of reasoning through explicit text tokens, Coconut uses special latent tokens to perform reasoning in a continuous hidden space. This repository provides tools to test the robustness of this approach under various noise conditions.

### Key Innovation

Traditional chain-of-thought reasoning happens in discrete token space, where each reasoning step is a text token. Coconut introduces **continuous latent reasoning** where thoughts occur in the model's hidden space between special `<|start-latent|>` and `<|end-latent|>` tokens. Noisy Coconut investigates:

- **How does noise in continuous thought space affect reasoning accuracy?**
- **Which types of noise are most disruptive?**
- **At what noise levels does the system break down?**

![Coconut Architecture](assets/coconut.png)

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/facebookresearch/coconut.git
cd coconut

# Setup environment
conda create --name coconut python=3.12
conda activate coconut
pip install -r requirements.txt

# Login to wandb (required for training)
wandb login
```

### Run Your First Noise Experiment

```bash
# Quick test with 10 questions and default Gaussian noise
python main.py --num-questions 10

# Test SNR (Signal-to-Noise Ratio) noise on 50 GSM8K questions
python main.py --noise-type snr --scales 10.0 5.0 2.0 1.0 0.5 --num-questions 50

# Comprehensive experiment with multiple noise types
python main.py --comprehensive --num-questions 100 --save-results
```

## 📊 Noise Types

Noisy Coconut supports 6 different noise injection strategies to understand robustness:

| Noise Type | Description | Best For |
|------------|-------------|----------|
| **gaussian** | Standard Gaussian N(0, σ²) | Baseline comparison |
| **gaussian_scaled** | Scaled relative to hidden state norm | Relative perturbation testing |
| **snr** | Signal-to-Noise Ratio controlled | Precise degradation curves |
| **uniform** | Uniform random noise | Non-Gaussian perturbations |
| **orthogonal** | Perpendicular to hidden state | Direction-independent effects |
| **targeted** | Along/opposite hidden state direction | Amplification/dampening |

See [NOISE_TYPES_GUIDE.md](NOISE_TYPES_GUIDE.md) for detailed explanations and usage examples.

## 🔬 Experimental Workflows

### 1. Noise Robustness Testing (Using Pre-trained Models)

Test how noise affects a Coconut model's reasoning:

```bash
# Test with Gaussian noise at multiple scales
python main.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --benchmark gsm8k \
  --noise-type gaussian \
  --scales 0.0 0.1 0.5 1.0 2.0 5.0 \
  --num-questions 100 \
  --save-results

# Test SNR-based noise (recommended for decay curves)
python main.py \
  --noise-type snr \
  --scales 100.0 50.0 20.0 10.0 5.0 2.0 1.0 0.5 0.1 \
  --save-results

# Test directional noise effects
python main.py \
  --noise-type targeted \
  --noise-direction opposite \
  --scales 0.1 0.3 0.5 0.7 1.0
```

### 2. Training Coconut Models

**Stage 0: Train Chain-of-Thought Baseline**

```bash
# Preprocess GSM8K dataset
bash preprocessing/gsm_icot.bash

# Train CoT model (4 GPUs)
torchrun --nnodes 1 --nproc_per_node 4 run.py args/gsm_cot.yaml
```

Expected validation accuracy: ~40%

**Stage 1+: Train Coconut with Continuous Thoughts**

```bash
# Update args/gsm_coconut.yaml with CoT checkpoint path
# Set: load_model_path: path/to/cot/checkpoint_X

# Train Coconut
torchrun --nnodes 1 --nproc_per_node 4 run.py args/gsm_coconut.yaml
```

Progressive training through stages 1, 2, 3+ with increasing latent capacity.

**Evaluation**

```bash
# Update args/gsm_coconut_eval.yaml with best checkpoint
# Set: load_model_path: path/to/coconut/checkpoint_best

torchrun --nnodes 1 --nproc_per_node 4 run.py args/gsm_coconut_eval.yaml
```

### 3. Custom Experiments

For advanced experimentation, use the underlying modules:

```python
from coconut import Coconut, apply_noise_to_hidden_states
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model
model = AutoModelForCausalLM.from_pretrained("your-model")
tokenizer = AutoTokenizer.from_pretrained("your-model")

# Add special tokens
tokenizer.add_special_tokens({
    'additional_special_tokens': ['<|latent|>', '<|start-latent|>', '<|end-latent|>']
})
model.resize_token_embeddings(len(tokenizer))

# Create Coconut wrapper with noise
coconut = Coconut(
    model,
    start_latent_id=tokenizer.convert_tokens_to_ids('<|start-latent|>'),
    end_latent_id=tokenizer.convert_tokens_to_ids('<|end-latent|>'),
    latent_token_id=tokenizer.convert_tokens_to_ids('<|latent|>'),
    eos_token_id=tokenizer.eos_token_id,
    c_thought=2,
    noise_scale=1.0,
    noise_type="snr"
)

# Generate with noisy continuous reasoning
outputs = coconut.generate(input_ids, max_new_tokens=512)
```

## 📁 Repository Structure

```
coconut/
├── main.py                  # 🆕 Main CLI for noise experiments
├── coconut.py              # Core Coconut model + noise injection
├── run.py                  # Training script (distributed)
├── dataset.py              # Dataset loading utilities
├── test.py                 # Legacy noise testing script
├── run_noisy_experiment.py # Legacy comprehensive experiments
├── NOISE_TYPES_GUIDE.md    # Detailed noise documentation
├── CLAUDE.md               # AI assistant development guide
├── README_original.md      # Original training-focused README
├── args/                   # Training configuration files
│   ├── gsm_cot.yaml
│   ├── gsm_coconut.yaml
│   └── *_eval.yaml
├── preprocessing/          # Dataset preprocessing
├── scripts/               # SLURM job scripts
├── data/                  # Training datasets
└── results/               # Experimental results
```

## 🎯 Benchmarks

Noisy Coconut supports multiple reasoning benchmarks:

- **GSM8K**: Grade school math reasoning (8.5K problems)
- **GSM-Symbolic**: Symbolic variant of GSM8K
- **MMLU**: Massive multitask language understanding

```bash
# Test on different benchmarks
python main.py --benchmark gsm8k --num-questions 100
python main.py --benchmark gsm-symbolic --num-questions 100
python main.py --benchmark mmlu --num-questions 500
```

## 📈 Results Analysis

Results are saved in JSON format with detailed metrics:

```json
{
  "timestamp": "2025-12-05T...",
  "model": "Qwen/Qwen2.5-0.5B-Instruct",
  "benchmark": "gsm8k",
  "experiments": [
    {
      "noise_type": "snr",
      "scales": {
        "10.0": {"accuracy": 0.85, "correct": 85, "total": 100},
        "5.0": {"accuracy": 0.78, "correct": 78, "total": 100},
        "1.0": {"accuracy": 0.42, "correct": 42, "total": 100}
      }
    }
  ]
}
```

## 🔧 Configuration

All training experiments use YAML configuration files in `args/`:

**Key Parameters:**
- `c_thought`: Number of continuous thoughts per reasoning step (default: 2)
- `epochs_per_stage`: Training epochs per stage (default: 3)
- `max_latent_stage`: Maximum training stages (default: 3)
- `uniform_prob`: Stage mixing probability for ablations (default: 0.0)

Example configuration:

```yaml
# args/gsm_coconut.yaml
project: coconut
save_path: /path/to/checkpoints
name: gsm-coconut

coconut: True
c_thought: 2
epochs_per_stage: 3
max_latent_stage: 3

model_id: openai-community/gpt2
load_model_path: /path/to/cot/checkpoint
train_path: data/gsm_train.json
val_path: data/gsm_valid.json

lr: 1e-4
batch_size_training: 32
```

## 🛠️ Advanced Usage

### Custom Datasets

Datasets must follow this JSON format:

```json
[
  {
    "question": "What is 2 + 2?",
    "answer": "4",
    "steps": ["First, note that 2 + 2", "equals 4"]
  }
]
```

### Distributed Training

```bash
# 4 GPUs (default)
torchrun --nnodes 1 --nproc_per_node 4 run.py args/config.yaml

# 8 GPUs
torchrun --nnodes 1 --nproc_per_node 8 run.py args/config.yaml
```

### Debug Mode

```bash
# Test with subset of data, no WandB logging
python main.py --num-questions 5
```

For training, set `debug: True` in config YAML.

## 📚 Citation

If you use this code or build upon the Coconut/Noisy Coconut approach, please cite:

```bibtex
@article{hao2024training,
  title={Training Large Language Models to Reason in a Continuous Latent Space},
  author={Hao, Shibo and Sukhbaatar, Sainbayar and Su, DiJia and Li, Xian and Hu, Zhiting and Weston, Jason and Tian, Yuandong},
  journal={arXiv preprint arXiv:2412.06769},
  year={2024}
}
```

## 🤝 Contributing

We welcome contributions! Please see:
- [CONTRIBUTING.md](CONTRIBUTING.md) - Contribution guidelines
- [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) - Community standards
- [CLAUDE.md](CLAUDE.md) - Development guide for AI assistants

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 🔗 Resources

- **Paper**: [Training Large Language Models to Reason in a Continuous Latent Space](https://arxiv.org/abs/2412.06769)
- **Original Training Guide**: See [README_original.md](README_original.md) for detailed training documentation
- **Noise Testing Guide**: [NOISE_TYPES_GUIDE.md](NOISE_TYPES_GUIDE.md)
- **Security Issues**: Use [Meta's bug bounty program](https://bugbounty.meta.com/)

## ⚡ Hardware Requirements

**For Noise Testing:**
- Minimum: 1 GPU with 16GB VRAM (or CPU for small models)
- Recommended: 1 x A100 40GB GPU

**For Training:**
- Default configs: 4 x A100 80GB GPUs
- Minimum: 1 GPU with 16GB VRAM (reduce batch size)
- Training time: 24-72 hours depending on dataset

## 🐛 Troubleshooting

**Issue: CUDA out of memory**
```bash
# Use smaller model or reduce batch size
python main.py --model Qwen/Qwen2.5-0.5B-Instruct --num-questions 20
```

**Issue: Dataset not loading**
```bash
# Preprocess datasets first
bash preprocessing/gsm_icot.bash
```

**Issue: WandB authentication**
```bash
wandb login
```

For more issues, check existing [GitHub issues](https://github.com/facebookresearch/coconut/issues).

---

**Made with 🥥 by Meta AI Research**
