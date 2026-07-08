# NoisyCoconut

**Counterfactual Consensus via Latent Space Reasoning**

![NoisyCoconut Diagram](./assets/noisy_coconut_diagram.png)

[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![OpenReview](https://img.shields.io/badge/OpenReview-TMLR-red)](https://openreview.net/forum?id=5aatZPiCv8)
[![arXiv](https://img.shields.io/badge/arXiv-2605.08221-b31b1b.svg)](https://arxiv.org/abs/2605.08221)

NoisyCoconut is a training-free inference-time method that enhances large language model (LLM) reliability by injecting controlled noise into latent representations to generate diverse reasoning paths. Agreement among these paths provides a confidence signal, enabling models to abstain when uncertain and achieve effective coverage-accuracy tradeoffs.

📄 **Paper**: [OpenReview (TMLR)](https://openreview.net/forum?id=5aatZPiCv8) | [arXiv](https://arxiv.org/abs/2605.08221)

## Key Features

- **No Retraining Required**: Operates directly on model representations during inference
- **Coverage-Accuracy Tradeoffs**: Enables selective prediction through agreement-based confidence estimation
- **Significant Error Reduction**: Unanimous agreement among noise-perturbed paths reduces error rates from 40-70% to below 15%
- **Model Agnostic**: Works across multiple LLM architectures (Qwen, Llama, Mixtral, DeepSeek, GPT-oss)

## How It Works

1. **Shared Forward Pass**: Run the LLM forward pass on the input to produce an initial hidden state h_0.
2. **Per-Branch Noise Injection**: Draw K independent noise vectors eta_1, ..., eta_K ~ N(0, sigma^2 * I) and add each to h_0, producing K distinct perturbed hidden states h_0^(i) = h_0 + eta_i (Eq. 1). Each branch receives its own unique perturbation — this is what causes paths to diverge.
3. **Latent Reasoning**: For each of the K perturbed variants, run the remaining T latent reasoning steps through the model's hidden states (Eq. 1–4).
4. **Path Diversity**: Pairwise trajectory diversity D_K (Eq. 5) measures how much the K paths diverged. Higher D_K indicates the paths explored meaningfully different regions of the solution space.
5. **Output Aggregation**: Decode each of the K paths autoregressively, then aggregate into a consensus answer (or abstain when agreement is insufficient, Eq. 6). Two aggregation modes are provided: majority voting (`run.py`) and logit probability-mass voting, which weights each path by the model's confidence in the tokens it generated (`run_logits.py`).

### Path Diversity Metric (D_K)

D_K quantifies how distinct the K reasoning trajectories are:

```
D_K = (2 / K(K-1)) * sum_{i<j} (1/T) * sum_{t=0}^{T-1} ||h_t^(i) - h_t^(j)||_2
```

A value of 0.0 means all branches collapsed to identical paths (no diversity). Higher values indicate the noise successfully induced genuinely different reasoning trajectories, which is required for agreement-based confidence estimation to be meaningful.

## Installation

Install as a package (recommended). This exposes the importable `noisycoconut`
library and the `noisycoconut-run`, `noisycoconut-logits`, and `noisycoconut-data`
console commands:

```bash
git clone https://github.com/mmjerge/noisycoconut.git
cd noisycoconut
pip install -e .
```

Alternatively, install only the dependencies and run the scripts in place:

```bash
pip install -r requirements.txt
```

### Requirements

- Python >= 3.9
- PyTorch >= 2.5
- Transformers >= 4.46
- mpmath >= 1.3 (high-precision logit probability-mass voting)
- CUDA-compatible GPU (recommended)

## Quick Start

### Download Benchmark Datasets

`data.py` (or `noisycoconut-data` after `pip install -e .`) downloads the
benchmarks. The available benchmarks are `gsm8k`, `gsm-symbolic`, `mmlu`,
`math`, `gpqa`, `gpqa-diamond`, and `gpqa-extended`.

```bash
# Download all benchmarks to ./data
python data.py            # or: noisycoconut-data

# Download to a specific directory
python data.py --data-dir ~/data/benchmarks

# Download only specific benchmarks
python data.py --benchmarks gsm8k mmlu math

# Force redownload existing files
python data.py --force

# Show stats about downloaded data
python data.py --stats
```

### Run Experiments

`run.py` defaults to `args/noisy-coconut.yaml`, so it runs with no arguments.
After `pip install -e .` you can equivalently use the `noisycoconut-run`
console command.

```bash
# Run with the default configuration (args/noisy-coconut.yaml)
python run.py                       # or: noisycoconut-run

# Use a specific config file
python run.py --config args/noisy-coconut.yaml

# Override configuration via CLI (OmegaConf dotlist)
python run.py experiment.num_questions=50 benchmark=math

# Run with a custom config file
python run.py --config my_config.yaml
```

### Logit Probability-Mass Voting

`run_logits.py` (or `noisycoconut-logits`) runs the same branching pipeline but
aggregates with logit probability-mass voting instead of plain majority voting.
Each branch is weighted by the exponentiated per-token log-probabilities of the
tokens it generated, computed at high precision via `mpmath`. It writes a
results file plus a companion `logits_*.json` file containing the per-branch,
per-token log-probability data.

```bash
python run_logits.py                # or: noisycoconut-logits
python run_logits.py --config args/noisy-coconut.yaml benchmark=gsm8k
```

## Configuration

Configuration is managed via YAML files. The default configuration is in `args/noisy-coconut.yaml`:

```yaml
# Options: gsm8k, gsm-symbolic, mmlu, math, gpqa, gpqa-diamond, gpqa-extended
benchmark: "mmlu"

model:
  name: "Qwen/Qwen2.5-7B-Instruct"
  max_new_tokens: 8192   # Large budget so long-reasoning models can finish

experiment:
  num_questions: 1000
  num_branches: 5        # K reasoning paths
  random_seed: 42

noise:
  scales: [0.2]          # Noise scale values to test
  type: "gaussian_scaled"  # Noise type (see table below)
  direction: null        # Direction for targeted noise
  schedule: "none"       # Per-step decay schedule: none, linear, cosine, exponential, adaptive
  lambda_decay: 0.5      # Exponential decay rate lambda (exponential/adaptive schedules)
  injection_step: 1      # Latent pass (1-8) at which noise is injected

sampling:
  temperature: 0.7
  top_p: 0.9

checkpoint:
  interval: 50           # Save progress every N questions

output_dir: "~/results"
```

### Noise Types

| Type | Description |
|------|-------------|
| `gaussian` | Standard Gaussian noise N(0, scale^2) |
| `gaussian_scaled` | Gaussian noise scaled to match hidden state norm |
| `snr` | Signal-to-Noise Ratio based noise |
| `uniform` | Uniform noise in [-scale, scale] |
| `orthogonal` | Noise orthogonal to hidden state direction |
| `targeted` | Noise in the direction of hidden state (amplifies/dampens) |
| `dropout` | Randomly zero out elements with probability = scale |

### Noise Schedules

`noise.schedule` controls how the per-step noise scale decays across the
post-branching latent passes. The default `none` reproduces the original
single-injection-at-branching-point behaviour.

| Schedule | Per-step scale |
|----------|----------------|
| `none` | No per-step noise (single injection at the branching point) |
| `linear` | sigma_t = sigma_0 * (1 - t / T) |
| `cosine` | sigma_t = sigma_0 * 0.5 * (1 + cos(pi * t / T)) |
| `exponential` | sigma_t = sigma_0 * exp(-lambda * t) |
| `adaptive` | sigma_t = sigma_0 * exp(-lambda * t) * (||h_t|| / mu_t) |

### Confidence Thresholds

- **Unanimous (5/5)**: Highest accuracy, lowest coverage
- **Strong Majority (4/5)**: High accuracy with moderate coverage
- **Moderate Majority (3/5)**: Balanced tradeoff
- **Minimal Plurality (2/5)**: Higher coverage, lower accuracy

## Benchmarks

The following benchmarks are supported:

- **GSM8K**: Grade-school math word problems
- **GSM-Symbolic**: Symbolic variant of GSM8K
- **MMLU**: Massive Multitask Language Understanding
- **MATH**: Hendrycks competition mathematics (500-question test split)
- **GPQA**: Graduate-level Google-proof Q&A (`gpqa`, `gpqa-diamond`, `gpqa-extended`)

## Project Structure

```
noisycoconut/
├── noisycoconut/             # Importable package
│   ├── __init__.py           # Public API (Coconut, apply_noise_to_hidden_states, ...)
│   └── coconut.py            # Core Coconut model with noise injection
├── run.py                    # Experiment runner: branching + majority voting
├── run_logits.py             # Runner with logit probability-mass voting
├── data.py                   # Dataset downloading and processing utilities
├── pyproject.toml            # Package metadata, dependencies, console scripts
├── requirements.txt          # Pinned dependencies for running in place
├── args/
│   └── noisy-coconut.yaml    # Default configuration
├── evaluate/                 # Analysis scripts and notebooks (D_K, self-consistency)
├── tests/
│   └── tests.py              # pytest test suite
└── assets/                   # Diagrams and images
```

## Core Components

### Coconut Model (`noisycoconut/coconut.py`)

The `Coconut` class wraps a base causal language model and implements continuous latent reasoning:

- **Latent Tokens**: Special `<|latent|>`, `<|start-latent|>`, and `<|end-latent|>` tokens mark reasoning regions
- **Latent Passes**: When start/end markers are adjacent, automatically performs 8 latent reasoning passes in continuous hidden state space
- **Noise Injection**: `apply_noise_to_hidden_states()` supports multiple noise distributions
- **Branching Generation**: `generate_with_branching()` creates K diverse paths with noise applied at a specified latent step (optionally returning the D_K diversity metrics)
- **Logit Scoring**: `generate_with_branching_logits()` mirrors the above and additionally returns the per-token log-probability of every sampled token for probability-mass voting

### Experiment Runners (`run.py`, `run_logits.py`)

Handle the full experimental pipeline:
- Model setup with special token registration
- Benchmark dataset loading (GSM8K, GSM-Symbolic, MMLU, MATH, GPQA)
- Branching generation with configurable noise
- Answer extraction and aggregation (majority voting in `run.py`, logit probability-mass voting in `run_logits.py`)
- Checkpoint/resume support with graceful SIGTERM/keyboard-interrupt handling
- Results aggregation and accuracy reporting

### Data Utilities (`data.py`)

Provides dataset downloading and preprocessing:
- Automatic download from HuggingFace datasets
- Consistent JSON format for all benchmarks
- Custom collation for latent token padding

## Python API

After `pip install -e .`, the core building blocks are importable:

```python
from noisycoconut import Coconut, apply_noise_to_hidden_states, compute_path_diversity
```

## Running Tests

```bash
# Run all tests
pytest tests/tests.py -v

# Run a specific test class
pytest tests/tests.py::TestApplyNoiseToHiddenStates -v

# Run with coverage
pytest tests/tests.py --cov=noisycoconut
```

## Limitations

- **Open-weight models only**: Requires access to internal model states
- **Computational overhead**: Generates K paths per query (linear scaling)
- **Discrete responses**: Best suited for tasks with well-defined answer agreement
- **Architecture sensitivity**: Some models (e.g., gpt-oss-20B) require modified configurations

## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{
jerge2026noisycoconut,
title={NoisyCoconut: Counterfactual Consensus via Latent Space Reasoning},
author={Michael M. Jerge and David Evans},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2026},
url={https://openreview.net/forum?id=5aatZPiCv8},
note={}
}
```

**Links:**
- 📝 [TMLR (OpenReview)](https://openreview.net/forum?id=5aatZPiCv8)
- 📚 [arXiv preprint](https://arxiv.org/abs/2605.08221)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This work builds on the Continuous Chain-of-Thought (Coconut) framework from Hao et al. (2025).
