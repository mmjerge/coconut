"""
Comprehensive Coconut noise robustness experiment on full GSM8K dataset.
Tests hypothesis: Accuracy decays logarithmically/exponentially with increasing noise.
Creates decay curve plots showing system breakdown.
"""

import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from coconut import Coconut
from datasets import load_dataset
from typing import List, Dict, Any, Tuple
import re
from scipy.optimize import curve_fit
from tqdm import tqdm
import os

def setup_model_and_tokenizer(model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"):
    """Initialize model and add special tokens."""
    print(f"Loading {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=True
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # Use FP32 for CPU
        device_map="cpu",
        trust_remote_code=True
    )
    
    special_tokens = {
        'additional_special_tokens': ['<|latent|>', '<|start-latent|>', '<|end-latent|>']
    }
    tokenizer.add_special_tokens(special_tokens)
    base_model.resize_token_embeddings(len(tokenizer))
    
    latent_token_id = tokenizer.convert_tokens_to_ids('<|latent|>')
    start_latent_id = tokenizer.convert_tokens_to_ids('<|start-latent|>')
    end_latent_id = tokenizer.convert_tokens_to_ids('<|end-latent|>')
    eos_token_id = tokenizer.eos_token_id
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Model loaded successfully!")
    
    return tokenizer, base_model, latent_token_id, start_latent_id, end_latent_id, eos_token_id


def load_gsm8k_dataset(num_questions: int = None) -> List[Dict[str, str]]:
    """Load GSM8K test dataset."""
    print(f"\nLoading GSM8K test set...")
    
    dataset = load_dataset("gsm8k", "main", split="test")
    
    if num_questions is None:
        num_questions = len(dataset)
    
    questions = []
    for i in range(min(num_questions, len(dataset))):
        item = dataset[i]
        questions.append({
            "question": item["question"],
            "answer": item["answer"],
            "question_id": i
        })
    
    print(f"Loaded {len(questions)} questions")
    return questions


def extract_numerical_answer(text: str) -> str:
    """Extract numerical answer from text."""
    patterns = [
        r'####\s*(\d+)',
        r'[Aa]nswer[:\s]+(\d+)',
        r'=\s*(\d+)',
        r'is\s+(\d+)',
        r'\$(\d+)',
        r'(\d+)\s*dollars?',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    
    numbers = re.findall(r'\d+', text)
    if numbers:
        return numbers[-1]
    
    return "NO_ANSWER_FOUND"


def create_coconut_input(tokenizer, question: str, start_latent_id: int, end_latent_id: int):
    """Create input for Coconut model."""
    prompt = f"Question: {question}\n\nPlease solve this step by step and provide the final numerical answer.\n\nAnswer:"
    
    question_ids = tokenizer.encode(prompt, return_tensors='pt', add_special_tokens=False)
    start_token = torch.tensor([[start_latent_id]])
    end_token = torch.tensor([[end_latent_id]])
    
    input_ids = torch.cat([question_ids, start_token, end_token], dim=1)
    return input_ids


def extract_generated_only(tokenizer, full_ids: torch.Tensor, original_input_ids: torch.Tensor, end_latent_id: int) -> str:
    """Extract only newly generated tokens after <end-latent>."""
    end_latent_positions = (original_input_ids[0] == end_latent_id).nonzero(as_tuple=True)[0]
    
    if len(end_latent_positions) > 0:
        end_latent_pos_expanded = end_latent_positions[0].item() + 8
        
        if full_ids.shape[1] > end_latent_pos_expanded + 1:
            generated_ids = full_ids[0, end_latent_pos_expanded + 1:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            return generated_text.strip()
    
    original_len = original_input_ids.shape[1]
    if full_ids.shape[1] > original_len:
        generated_ids = full_ids[0, original_len:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        return generated_text.strip()
    
    return ""


def test_question_with_noise(
    coconut_model,
    tokenizer,
    question: str,
    noise_scale: float,
    start_latent_id: int,
    end_latent_id: int,
    max_new_tokens: int = 100,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """Test a single question with specific noise scale."""
    
    coconut_model.eval()
    
    input_ids = create_coconut_input(tokenizer, question, start_latent_id, end_latent_id)
    original_input_ids = input_ids.clone()
    input_ids = input_ids.to(device)
    
    with torch.no_grad():
        try:
            generated_ids = coconut_model.generate(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=max_new_tokens,
                noise_scale=noise_scale
            )
            
            generated_text = extract_generated_only(
                tokenizer, 
                generated_ids, 
                original_input_ids, 
                end_latent_id
            )
            
            return {
                "success": True,
                "generated_text": generated_text,
                "error": None
            }
            
        except Exception as e:
            return {
                "success": False,
                "generated_text": "",
                "error": str(e)
            }


def run_full_experiment(
    num_questions: int = None,  # None = full dataset
    noise_scales: List[float] = None,
    max_new_tokens: int = 100,
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    save_interval: int = 50,
    output_dir: str = "coconut_results"
) -> Dict[str, Any]:
    """Run complete experiment on full GSM8K."""
    
    if noise_scales is None:
        noise_scales = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    
    os.makedirs(output_dir, exist_ok=True)
    device = 'cpu'
    
    print("="*70)
    print("COCONUT FULL GSM8K NOISE EXPERIMENT")
    print("="*70)
    print(f"Noise scales: {noise_scales}")
    print(f"Device: {device}")
    print(f"Model: {model_name}")
    
    # Setup
    tokenizer, base_model, latent_token_id, start_latent_id, end_latent_id, eos_token_id = setup_model_and_tokenizer(model_name)
    
    coconut_model = Coconut(
        base_causallm=base_model,
        latent_token_id=latent_token_id,
        start_latent_id=start_latent_id,
        end_latent_id=end_latent_id,
        eos_token_id=eos_token_id
    )
    
    # Load questions
    questions = load_gsm8k_dataset(num_questions)
    
    # Initialize results structure
    results = {
        "experiment_info": {
            "timestamp": datetime.now().isoformat(),
            "num_questions": len(questions),
            "noise_scales": noise_scales,
            "max_new_tokens": max_new_tokens,
            "device": device,
            "model": model_name,
            "latent_passes": 8
        },
        "accuracy_by_noise": {},  # Will store {noise_scale: accuracy}
        "questions": []
    }
    
    # Initialize accuracy tracking
    for noise_scale in noise_scales:
        results["accuracy_by_noise"][str(noise_scale)] = {
            "correct": 0,
            "total": 0,
            "accuracy": 0.0
        }
    
    # Test each question
    print(f"\nTesting {len(questions)} questions across {len(noise_scales)} noise levels...")
    
    for q_idx, q_data in enumerate(tqdm(questions, desc="Processing questions")):
        question_results = {
            "question_id": q_data["question_id"],
            "question": q_data["question"],
            "expected_answer": q_data["answer"],
            "noise_tests": []
        }
        
        for noise_scale in noise_scales:
            result = test_question_with_noise(
                coconut_model=coconut_model,
                tokenizer=tokenizer,
                question=q_data["question"],
                noise_scale=noise_scale,
                start_latent_id=start_latent_id,
                end_latent_id=end_latent_id,
                max_new_tokens=max_new_tokens,
                device=device
            )
            
            if result["success"]:
                generated_answer = extract_numerical_answer(result["generated_text"])
                expected_answer = extract_numerical_answer(q_data["answer"])
                is_correct = generated_answer == expected_answer
                
                # Update accuracy tracking
                results["accuracy_by_noise"][str(noise_scale)]["total"] += 1
                if is_correct:
                    results["accuracy_by_noise"][str(noise_scale)]["correct"] += 1
                
                question_results["noise_tests"].append({
                    "noise_scale": noise_scale,
                    "generated_answer": generated_answer,
                    "expected_answer": expected_answer,
                    "is_correct": is_correct,
                    "success": True
                })
            else:
                results["accuracy_by_noise"][str(noise_scale)]["total"] += 1
                question_results["noise_tests"].append({
                    "noise_scale": noise_scale,
                    "generated_answer": None,
                    "expected_answer": extract_numerical_answer(q_data["answer"]),
                    "is_correct": False,
                    "success": False,
                    "error": result["error"]
                })
        
        results["questions"].append(question_results)
        
        # Save intermediate results
        if (q_idx + 1) % save_interval == 0:
            # Calculate accuracies
            for noise_scale in noise_scales:
                ns_str = str(noise_scale)
                if results["accuracy_by_noise"][ns_str]["total"] > 0:
                    results["accuracy_by_noise"][ns_str]["accuracy"] = (
                        results["accuracy_by_noise"][ns_str]["correct"] / 
                        results["accuracy_by_noise"][ns_str]["total"] * 100
                    )
            
            checkpoint_file = os.path.join(output_dir, f"checkpoint_{q_idx+1}.json")
            with open(checkpoint_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nSaved checkpoint at question {q_idx + 1}")
    
    # Final accuracy calculation
    for noise_scale in noise_scales:
        ns_str = str(noise_scale)
        if results["accuracy_by_noise"][ns_str]["total"] > 0:
            results["accuracy_by_noise"][ns_str]["accuracy"] = (
                results["accuracy_by_noise"][ns_str]["correct"] / 
                results["accuracy_by_noise"][ns_str]["total"] * 100
            )
    
    return results


def fit_decay_curves(noise_scales: np.ndarray, accuracies: np.ndarray) -> Dict[str, Any]:
    """Fit different decay models to the data."""
    
    # Exponential decay: y = a * exp(-b * x)
    def exponential_decay(x, a, b):
        return a * np.exp(-b * x)
    
    # Inverse sigmoid: y = a / (1 + b * x^c)
    def inverse_sigmoid(x, a, b, c):
        return a / (1 + b * np.power(x, c))
    
    # Linear decay: y = a - b * x
    def linear_decay(x, a, b):
        return a - b * x
    
    fits = {}
    
    try:
        # Exponential fit
        popt_exp, _ = curve_fit(exponential_decay, noise_scales, accuracies, 
                                 p0=[100, 0.1], maxfev=10000)
        fits['exponential'] = {
            'params': popt_exp,
            'func': lambda x: exponential_decay(x, *popt_exp),
            'name': f'Exponential: {popt_exp[0]:.1f} * exp(-{popt_exp[1]:.3f} * x)'
        }
    except:
        print("Exponential fit failed")
    
    try:
        # Inverse sigmoid fit
        popt_inv, _ = curve_fit(inverse_sigmoid, noise_scales, accuracies,
                                p0=[100, 1, 1], maxfev=10000)
        fits['inverse_sigmoid'] = {
            'params': popt_inv,
            'func': lambda x: inverse_sigmoid(x, *popt_inv),
            'name': f'Inverse Sigmoid: {popt_inv[0]:.1f} / (1 + {popt_inv[1]:.3f} * x^{popt_inv[2]:.2f})'
        }
    except:
        print("Inverse sigmoid fit failed")
    
    try:
        # Linear fit
        popt_lin, _ = curve_fit(linear_decay, noise_scales, accuracies,
                               p0=[100, 10])
        fits['linear'] = {
            'params': popt_lin,
            'func': lambda x: linear_decay(x, *popt_lin),
            'name': f'Linear: {popt_lin[0]:.1f} - {popt_lin[1]:.2f} * x'
        }
    except:
        print("Linear fit failed")
    
    return fits


def plot_decay_curves(results: Dict[str, Any], output_dir: str = "coconut_results"):
    """Create decay curve plots."""
    
    # Extract data
    noise_scales = []
    accuracies = []
    
    for ns_str, data in sorted(results["accuracy_by_noise"].items(), key=lambda x: float(x[0])):
        noise_scales.append(float(ns_str))
        accuracies.append(data["accuracy"])
    
    noise_scales = np.array(noise_scales)
    accuracies = np.array(accuracies)
    
    # Fit curves
    fits = fit_decay_curves(noise_scales, accuracies)
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Plot actual data
    plt.plot(noise_scales, accuracies, 'o-', linewidth=2, markersize=10, 
             label='Actual Accuracy', color='red', zorder=5)
    
    # Plot fitted curves
    x_smooth = np.linspace(0, max(noise_scales), 1000)
    colors = ['blue', 'green', 'purple']
    
    for (fit_name, fit_data), color in zip(fits.items(), colors):
        y_fit = fit_data['func'](x_smooth)
        plt.plot(x_smooth, y_fit, '--', linewidth=2, label=fit_data['name'], 
                color=color, alpha=0.7)
    
    plt.xlabel('Noise Scale', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    plt.title('Coconut System Breakdown: Accuracy vs Noise Scale\n' + 
              f"Model: {results['experiment_info']['model']}, " +
              f"Questions: {results['experiment_info']['num_questions']}", 
              fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11, loc='best')
    plt.xlim(0, max(noise_scales))
    plt.ylim(0, max(100, max(accuracies) * 1.1))
    
    # Add annotation for breakdown point
    if len(accuracies) > 1:
        # Find where accuracy drops below 50%
        breakdown_idx = np.where(accuracies < 50)[0]
        if len(breakdown_idx) > 0:
            breakdown_noise = noise_scales[breakdown_idx[0]]
            plt.axvline(x=breakdown_noise, color='red', linestyle=':', linewidth=2, alpha=0.5)
            plt.text(breakdown_noise, 50, f'  Breakdown\n  at noise={breakdown_noise}', 
                    fontsize=10, color='red', fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    plot_file = os.path.join(output_dir, "accuracy_decay_curve.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {plot_file}")
    
    plt.show()


def print_summary(results: Dict[str, Any]):
    """Print experiment summary."""
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)
    
    print(f"\nModel: {results['experiment_info']['model']}")
    print(f"Total Questions: {results['experiment_info']['num_questions']}")
    print(f"Latent Passes: {results['experiment_info']['latent_passes']}")
    
    print("\n" + "-"*70)
    print(f"{'Noise Scale':<15} {'Correct':<10} {'Total':<10} {'Accuracy':<10}")
    print("-"*70)
    
    for ns_str, data in sorted(results["accuracy_by_noise"].items(), key=lambda x: float(x[0])):
        noise = float(ns_str)
        correct = data["correct"]
        total = data["total"]
        accuracy = data["accuracy"]
        print(f"{noise:<15.2f} {correct:<10} {total:<10} {accuracy:>7.2f}%")


def main():
    """Main experiment runner."""
    
    # Configuration
    NUM_QUESTIONS = None  # None = full dataset (1319 questions)
    # Or set to a smaller number for testing: NUM_QUESTIONS = 100
    
    NOISE_SCALES = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    MAX_NEW_TOKENS = 100
    MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
    OUTPUT_DIR = "coconut_full_results"
    SAVE_INTERVAL = 50  # Save checkpoint every N questions
    
    # Run experiment
    results = run_full_experiment(
        num_questions=NUM_QUESTIONS,
        noise_scales=NOISE_SCALES,
        max_new_tokens=MAX_NEW_TOKENS,
        model_name=MODEL_NAME,
        save_interval=SAVE_INTERVAL,
        output_dir=OUTPUT_DIR
    )
    
    # Save final results
    final_file = os.path.join(OUTPUT_DIR, "final_results.json")
    with open(final_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nFinal results saved to: {final_file}")
    
    # Print summary
    print_summary(results)
    
    # Plot decay curves
    plot_decay_curves(results, OUTPUT_DIR)
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()