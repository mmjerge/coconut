#!/usr/bin/env python3
"""
Noisy Coconut: Test noise robustness of continuous latent reasoning models.

This is the main entry point for testing Coconut models with various noise types
and levels to understand how noise affects continuous reasoning performance.

Usage:
    # Quick test with default settings
    python main.py

    # Test specific noise type on GSM8K
    python main.py --noise-type snr --scales 10.0 5.0 2.0 1.0 0.5 --num-questions 100

    # Comprehensive experiment
    python main.py --noise-type gaussian --benchmark gsm8k --num-questions 500 --save-results

    # Test multiple noise types
    python main.py --noise-type gaussian gaussian_scaled snr --comprehensive

For detailed noise type documentation, see NOISE_TYPES_GUIDE.md
"""

import argparse
import json
import torch
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from coconut import Coconut, apply_noise_to_hidden_states
from datasets import load_dataset
from tqdm import tqdm
import re


# ============================================================================
# Model Setup
# ============================================================================

def setup_coconut_model(
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    device: str = "auto"
) -> tuple:
    """
    Initialize Coconut model with special tokens.

    Args:
        model_name: HuggingFace model ID
        device: Device placement ("auto", "cuda", "cpu")

    Returns:
        (tokenizer, model, latent_token_id, start_latent_id, end_latent_id, eos_token_id)
    """
    print(f"Loading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=True
    )

    # Determine dtype and device
    if device == "auto":
        if torch.cuda.is_available():
            dtype = torch.bfloat16
            device_map = "auto"
        else:
            dtype = torch.float32
            device_map = "cpu"
    elif device == "cuda":
        dtype = torch.bfloat16
        device_map = "cuda"
    else:
        dtype = torch.float32
        device_map = "cpu"

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True
    )

    # Add special Coconut tokens
    special_tokens = {
        'additional_special_tokens': ['<|latent|>', '<|start-latent|>', '<|end-latent|>']
    }
    num_added = tokenizer.add_special_tokens(special_tokens)
    base_model.resize_token_embeddings(len(tokenizer))

    # Get token IDs
    latent_token_id = tokenizer.convert_tokens_to_ids('<|latent|>')
    start_latent_id = tokenizer.convert_tokens_to_ids('<|start-latent|>')
    end_latent_id = tokenizer.convert_tokens_to_ids('<|end-latent|>')
    eos_token_id = tokenizer.eos_token_id

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"✓ Model loaded successfully")
    print(f"  Added {num_added} special tokens")
    print(f"  Device: {device_map}, dtype: {dtype}")

    return tokenizer, base_model, latent_token_id, start_latent_id, end_latent_id, eos_token_id


# ============================================================================
# Dataset Loading
# ============================================================================

def load_benchmark(benchmark: str, num_questions: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Load benchmark dataset.

    Args:
        benchmark: One of "gsm8k", "gsm-symbolic", "mmlu"
        num_questions: Number of questions to load (None = all)

    Returns:
        List of question dictionaries
    """
    print(f"\nLoading {benchmark.upper()} dataset...")

    if benchmark == "gsm8k":
        dataset = load_dataset("gsm8k", "main", split="test")
    elif benchmark == "gsm-symbolic":
        dataset = load_dataset("apple/gsm-symbolic", split="test", trust_remote_code=True)
    elif benchmark == "mmlu":
        dataset = load_dataset("cais/mmlu", "all", split="test", trust_remote_code=True)
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")

    questions = []
    limit = min(num_questions or len(dataset), len(dataset))

    for i in range(limit):
        item = dataset[i]
        questions.append({
            "question": item["question"],
            "answer": item["answer"],
            "question_id": i,
            "benchmark": benchmark
        })

    print(f"✓ Loaded {len(questions)} questions")
    return questions


# ============================================================================
# Answer Extraction
# ============================================================================

def extract_answer(text: str) -> str:
    """Extract numerical answer from generated text."""
    def clean_number(num_str: str) -> str:
        return num_str.replace(',', '').replace('$', '').strip()

    patterns = [
        r'\\boxed\{([\d,]+)\}',
        r'####\s*([\d,]+)',
        r'[Tt]he final answer is:?\s*\$?\s*([\d,]+)',
        r'[Ff]inal answer:?\s*\$?\s*([\d,]+)',
        r'[Aa]nswer:?\s*\$?\s*([\d,]+)',
        r'=\s*\$?\s*([\d,]+)',
    ]

    for pattern in patterns:
        matches = list(re.finditer(pattern, text))
        if matches:
            return clean_number(matches[-1].group(1))

    # Fallback: return last number
    numbers = re.findall(r'[\d,]+', text)
    if numbers:
        cleaned = [clean_number(n) for n in numbers if clean_number(n)]
        if cleaned:
            return cleaned[-1]

    return "NO_ANSWER"


# ============================================================================
# Coconut Inference
# ============================================================================

def create_coconut_input(
    tokenizer,
    question: str,
    start_latent_id: int,
    end_latent_id: int
) -> torch.Tensor:
    """Create input with latent reasoning markers."""
    messages = [
        {"role": "user", "content": f"{question}\n\nPlease solve this step by step."}
    ]

    question_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors='pt',
        add_special_tokens=True
    )

    # Add latent markers
    start_token = torch.tensor([[start_latent_id]])
    end_token = torch.tensor([[end_latent_id]])
    input_ids = torch.cat([question_ids, start_token, end_token], dim=1)

    return input_ids


def run_noisy_inference(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    end_latent_id: int,
    noise_type: str = "gaussian",
    noise_scale: float = 0.0,
    noise_direction: Optional[str] = None,
    max_new_tokens: int = 512
) -> tuple:
    """
    Run inference with noise injection in continuous latent space.

    Returns:
        (generated_text, noise_info)
    """
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    # Create Coconut wrapper
    coconut_model = Coconut(
        model,
        start_latent_id=tokenizer.convert_tokens_to_ids('<|start-latent|>'),
        end_latent_id=end_latent_id,
        latent_token_id=tokenizer.convert_tokens_to_ids('<|latent|>'),
        eos_token_id=tokenizer.eos_token_id,
        c_thought=2,  # Number of continuous thoughts
        noise_scale=noise_scale,
        noise_type=noise_type,
        noise_direction=noise_direction
    )

    # Generate
    with torch.no_grad():
        output_ids = coconut_model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )

    # Extract only generated portion (after end_latent_id)
    end_latent_positions = (output_ids[0] == end_latent_id).nonzero(as_tuple=True)[0]
    if len(end_latent_positions) > 0:
        start_idx = end_latent_positions[0].item() + 1
        generated_ids = output_ids[0, start_idx:]
    else:
        generated_ids = output_ids[0, input_ids.shape[1]:]

    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Get noise info if available
    noise_info = getattr(coconut_model, 'last_noise_info', {})

    return generated_text, noise_info


# ============================================================================
# Experiment Runner
# ============================================================================

def run_experiment(
    model,
    tokenizer,
    questions: List[Dict],
    noise_type: str,
    noise_scales: List[float],
    start_latent_id: int,
    end_latent_id: int,
    noise_direction: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run noise robustness experiment across multiple noise scales.

    Returns:
        Dictionary with results for each noise scale
    """
    results = {
        "noise_type": noise_type,
        "noise_direction": noise_direction,
        "scales": {}
    }

    for scale in noise_scales:
        print(f"\n{'='*60}")
        print(f"Testing {noise_type} with scale={scale}")
        print(f"{'='*60}")

        correct = 0
        total = 0
        scale_results = []

        for q in tqdm(questions, desc=f"Scale {scale}"):
            # Create input
            input_ids = create_coconut_input(
                tokenizer,
                q["question"],
                start_latent_id,
                end_latent_id
            )

            # Generate with noise
            generated_text, noise_info = run_noisy_inference(
                model,
                tokenizer,
                input_ids,
                end_latent_id,
                noise_type=noise_type,
                noise_scale=scale,
                noise_direction=noise_direction
            )

            # Extract answer
            predicted = extract_answer(generated_text)
            true_answer = extract_answer(q["answer"])
            is_correct = (predicted == true_answer)

            if is_correct:
                correct += 1
            total += 1

            scale_results.append({
                "question_id": q["question_id"],
                "predicted": predicted,
                "true_answer": true_answer,
                "correct": is_correct,
                "generated_text": generated_text[:200]  # Truncate for storage
            })

        accuracy = correct / total if total > 0 else 0.0
        print(f"\n✓ Scale {scale}: Accuracy = {accuracy:.2%} ({correct}/{total})")

        results["scales"][str(scale)] = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "details": scale_results
        }

    return results


# ============================================================================
# Main CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Noisy Coconut: Test noise robustness of continuous reasoning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test
  python main.py --num-questions 10

  # SNR noise experiment
  python main.py --noise-type snr --scales 10.0 5.0 2.0 1.0 0.5

  # Comprehensive multi-noise test
  python main.py --comprehensive --num-questions 100 --save-results
        """
    )

    # Model settings
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct",
                       help="HuggingFace model ID")
    parser.add_argument("--device", default="auto",
                       choices=["auto", "cuda", "cpu"],
                       help="Device placement")

    # Dataset settings
    parser.add_argument("--benchmark", default="gsm8k",
                       choices=["gsm8k", "gsm-symbolic", "mmlu"],
                       help="Benchmark dataset")
    parser.add_argument("--num-questions", type=int, default=50,
                       help="Number of questions to test")

    # Noise settings
    parser.add_argument("--noise-type", nargs="+",
                       default=["gaussian"],
                       choices=["gaussian", "gaussian_scaled", "snr", "uniform",
                               "orthogonal", "targeted"],
                       help="Noise type(s) to test")
    parser.add_argument("--scales", nargs="+", type=float,
                       default=[0.0, 0.1, 0.5, 1.0, 2.0],
                       help="Noise scales to test")
    parser.add_argument("--noise-direction",
                       choices=["same", "opposite", "random_orthogonal"],
                       help="Direction for targeted/orthogonal noise")

    # Experiment settings
    parser.add_argument("--comprehensive", action="store_true",
                       help="Run comprehensive experiment with multiple noise types")
    parser.add_argument("--save-results", action="store_true",
                       help="Save results to JSON file")
    parser.add_argument("--output-dir", default="results/noisy_coconut",
                       help="Output directory for results")

    args = parser.parse_args()

    # Setup
    print("\n" + "="*60)
    print("NOISY COCONUT - Noise Robustness Testing")
    print("="*60)

    tokenizer, model, latent_id, start_id, end_id, eos_id = setup_coconut_model(
        args.model, args.device
    )

    questions = load_benchmark(args.benchmark, args.num_questions)

    # Comprehensive mode: test all noise types
    if args.comprehensive:
        noise_types = ["gaussian", "gaussian_scaled", "snr", "uniform", "orthogonal"]
        print(f"\n🔬 Running comprehensive experiment with {len(noise_types)} noise types")
    else:
        noise_types = args.noise_type

    # Run experiments
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "benchmark": args.benchmark,
        "num_questions": len(questions),
        "experiments": []
    }

    for noise_type in noise_types:
        results = run_experiment(
            model,
            tokenizer,
            questions,
            noise_type,
            args.scales,
            start_id,
            end_id,
            args.noise_direction
        )
        all_results["experiments"].append(results)

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for exp in all_results["experiments"]:
        print(f"\nNoise Type: {exp['noise_type']}")
        for scale_str, data in exp["scales"].items():
            print(f"  Scale {scale_str:>6}: {data['accuracy']:.2%}")

    # Save results
    if args.save_results:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"results_{timestamp}.json"

        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)

        print(f"\n✓ Results saved to: {output_file}")

    print("\n" + "="*60)
    print("Experiment complete!")
    print("="*60)


if __name__ == "__main__":
    main()
