"""
Multi-GPU Coconut noise robustness experiment using Accelerate.
Distributes GSM8K questions across multiple GPUs for parallel testing.
"""

import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from coconut import Coconut
from datasets import load_dataset
from typing import List, Dict, Any
import re
from scipy.optimize import curve_fit
from tqdm import tqdm
import os
from accelerate import Accelerator
from accelerate.utils import gather_object

def setup_model_and_tokenizer(model_name: str, accelerator):
    """Initialize model and add special tokens."""
    accelerator.print(f"Loading {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=True
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # FP16 for GPU efficiency
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
    
    accelerator.print(f"Model loaded on {accelerator.device}")
    
    return tokenizer, base_model, latent_token_id, start_latent_id, end_latent_id, eos_token_id


def load_gsm8k_dataset(num_questions: int = None) -> List[Dict[str, str]]:
    """Load GSM8K test dataset."""
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
    device: str = 'cuda'
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


def process_questions_on_gpu(
    questions_subset: List[Dict],
    noise_scales: List[float],
    coconut_model,
    tokenizer,
    start_latent_id: int,
    end_latent_id: int,
    max_new_tokens: int,
    device: str,
    gpu_id: int
) -> List[Dict]:
    """Process a subset of questions on a single GPU."""
    
    results = []
    
    for q_data in tqdm(questions_subset, desc=f"GPU {gpu_id}", position=gpu_id):
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
                
                question_results["noise_tests"].append({
                    "noise_scale": noise_scale,
                    "generated_answer": generated_answer,
                    "expected_answer": expected_answer,
                    "is_correct": is_correct,
                    "success": True
                })
            else:
                question_results["noise_tests"].append({
                    "noise_scale": noise_scale,
                    "generated_answer": None,
                    "expected_answer": extract_numerical_answer(q_data["answer"]),
                    "is_correct": False,
                    "success": False,
                    "error": result["error"]
                })
        
        results.append(question_results)
    
    return results


def run_distributed_experiment(
    num_questions: int = None,
    noise_scales: List[float] = None,
    max_new_tokens: int = 100,
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    output_dir: str = "coconut_results"
):
    """Run experiment distributed across multiple GPUs."""
    
    # Initialize accelerator
    accelerator = Accelerator()
    
    if noise_scales is None:
        noise_scales = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Print config on main process only
    if accelerator.is_main_process:
        print("="*70)
        print("COCONUT MULTI-GPU EXPERIMENT")
        print("="*70)
        print(f"Number of GPUs: {accelerator.num_processes}")
        print(f"Noise scales: {noise_scales}")
        print(f"Model: {model_name}")
    
    # Load model and tokenizer
    tokenizer, base_model, latent_token_id, start_latent_id, end_latent_id, eos_token_id = setup_model_and_tokenizer(
        model_name, accelerator
    )
    
    # Create Coconut wrapper
    coconut_model = Coconut(
        base_causallm=base_model,
        latent_token_id=latent_token_id,
        start_latent_id=start_latent_id,
        end_latent_id=end_latent_id,
        eos_token_id=eos_token_id
    )
    
    # Prepare model with accelerator
    coconut_model = accelerator.prepare(coconut_model)
    
    # Load questions on main process
    if accelerator.is_main_process:
        questions = load_gsm8k_dataset(num_questions)
        print(f"Loaded {len(questions)} questions")
    else:
        questions = None
    
    # Broadcast questions to all processes
    questions = accelerator.broadcast_object_list([questions])[0]
    
    # Split questions across GPUs
    questions_per_gpu = len(questions) // accelerator.num_processes
    start_idx = accelerator.process_index * questions_per_gpu
    
    if accelerator.process_index == accelerator.num_processes - 1:
        # Last GPU gets remaining questions
        end_idx = len(questions)
    else:
        end_idx = start_idx + questions_per_gpu
    
    questions_subset = questions[start_idx:end_idx]
    
    accelerator.print(f"GPU {accelerator.process_index}: Processing questions {start_idx} to {end_idx-1} ({len(questions_subset)} total)")
    
    # Process questions on this GPU
    local_results = process_questions_on_gpu(
        questions_subset=questions_subset,
        noise_scales=noise_scales,
        coconut_model=coconut_model,
        tokenizer=tokenizer,
        start_latent_id=start_latent_id,
        end_latent_id=end_latent_id,
        max_new_tokens=max_new_tokens,
        device=accelerator.device,
        gpu_id=accelerator.process_index
    )
    
    # Gather results from all GPUs
    accelerator.wait_for_everyone()
    all_results = gather_object(local_results)
    
    # Main process aggregates and saves results
    if accelerator.is_main_process:
        # Flatten gathered results
        all_questions_results = []
        for result_list in all_results:
            all_questions_results.extend(result_list)
        
        # Sort by question_id to maintain order
        all_questions_results.sort(key=lambda x: x["question_id"])
        
        # Calculate accuracy by noise scale
        accuracy_by_noise = {}
        for noise_scale in noise_scales:
            accuracy_by_noise[str(noise_scale)] = {
                "correct": 0,
                "total": 0,
                "accuracy": 0.0
            }
        
        for q_result in all_questions_results:
            for test in q_result["noise_tests"]:
                ns_str = str(test["noise_scale"])
                accuracy_by_noise[ns_str]["total"] += 1
                if test["success"] and test["is_correct"]:
                    accuracy_by_noise[ns_str]["correct"] += 1
        
        # Calculate percentages
        for ns_str in accuracy_by_noise:
            if accuracy_by_noise[ns_str]["total"] > 0:
                accuracy_by_noise[ns_str]["accuracy"] = (
                    accuracy_by_noise[ns_str]["correct"] / 
                    accuracy_by_noise[ns_str]["total"] * 100
                )
        
        # Create final results structure
        results = {
            "experiment_info": {
                "timestamp": datetime.now().isoformat(),
                "num_questions": len(all_questions_results),
                "noise_scales": noise_scales,
                "max_new_tokens": max_new_tokens,
                "num_gpus": accelerator.num_processes,
                "model": model_name,
                "latent_passes": 8
            },
            "accuracy_by_noise": accuracy_by_noise,
            "questions": all_questions_results
        }
        
        # Save results
        final_file = os.path.join(output_dir, "final_results.json")
        with open(final_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {final_file}")
        
        # Print summary
        print_summary(results)
        
        # Plot
        plot_decay_curves(results, output_dir)
        
        print("\n" + "="*70)
        print("EXPERIMENT COMPLETE!")
        print("="*70)
        
        return results
    
    return None


def fit_decay_curves(noise_scales: np.ndarray, accuracies: np.ndarray) -> Dict[str, Any]:
    """Fit different decay models to the data."""
    
    def exponential_decay(x, a, b):
        return a * np.exp(-b * x)
    
    def inverse_sigmoid(x, a, b, c):
        return a / (1 + b * np.power(x, c))
    
    def linear_decay(x, a, b):
        return a - b * x
    
    fits = {}
    
    try:
        popt_exp, _ = curve_fit(exponential_decay, noise_scales, accuracies, 
                                 p0=[100, 0.1], maxfev=10000)
        fits['exponential'] = {
            'params': popt_exp,
            'func': lambda x: exponential_decay(x, *popt_exp),
            'name': f'Exponential: {popt_exp[0]:.1f} * exp(-{popt_exp[1]:.3f} * x)'
        }
    except:
        pass
    
    try:
        popt_inv, _ = curve_fit(inverse_sigmoid, noise_scales, accuracies,
                                p0=[100, 1, 1], maxfev=10000)
        fits['inverse_sigmoid'] = {
            'params': popt_inv,
            'func': lambda x: inverse_sigmoid(x, *popt_inv),
            'name': f'Inverse Sigmoid: {popt_inv[0]:.1f} / (1 + {popt_inv[1]:.3f} * x^{popt_inv[2]:.2f})'
        }
    except:
        pass
    
    try:
        popt_lin, _ = curve_fit(linear_decay, noise_scales, accuracies,
                               p0=[100, 10])
        fits['linear'] = {
            'params': popt_lin,
            'func': lambda x: linear_decay(x, *popt_lin),
            'name': f'Linear: {popt_lin[0]:.1f} - {popt_lin[1]:.2f} * x'
        }
    except:
        pass
    
    return fits


def plot_decay_curves(results: Dict[str, Any], output_dir: str):
    """Create decay curve plots."""
    
    noise_scales = []
    accuracies = []
    
    for ns_str, data in sorted(results["accuracy_by_noise"].items(), key=lambda x: float(x[0])):
        noise_scales.append(float(ns_str))
        accuracies.append(data["accuracy"])
    
    noise_scales = np.array(noise_scales)
    accuracies = np.array(accuracies)
    
    fits = fit_decay_curves(noise_scales, accuracies)
    
    plt.figure(figsize=(12, 8))
    
    plt.plot(noise_scales, accuracies, 'o-', linewidth=2, markersize=10, 
             label='Actual Accuracy', color='red', zorder=5)
    
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
              f"Questions: {results['experiment_info']['num_questions']}, " +
              f"GPUs: {results['experiment_info']['num_gpus']}", 
              fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11, loc='best')
    plt.xlim(0, max(noise_scales))
    plt.ylim(0, max(100, max(accuracies) * 1.1))
    
    if len(accuracies) > 1:
        breakdown_idx = np.where(accuracies < 50)[0]
        if len(breakdown_idx) > 0:
            breakdown_noise = noise_scales[breakdown_idx[0]]
            plt.axvline(x=breakdown_noise, color='red', linestyle=':', linewidth=2, alpha=0.5)
            plt.text(breakdown_noise, 50, f'  Breakdown\n  at noise={breakdown_noise}', 
                    fontsize=10, color='red', fontweight='bold')
    
    plt.tight_layout()
    
    plot_file = os.path.join(output_dir, "accuracy_decay_curve.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {plot_file}")


def print_summary(results: Dict[str, Any]):
    """Print experiment summary."""
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)
    
    print(f"\nModel: {results['experiment_info']['model']}")
    print(f"GPUs Used: {results['experiment_info']['num_gpus']}")
    print(f"Total Questions: {results['experiment_info']['num_questions']}")
    
    print("\n" + "-"*70)
    print(f"{'Noise Scale':<15} {'Correct':<10} {'Total':<10} {'Accuracy':<10}")
    print("-"*70)
    
    for ns_str, data in sorted(results["accuracy_by_noise"].items(), key=lambda x: float(x[0])):
        noise = float(ns_str)
        print(f"{noise:<15.2f} {data['correct']:<10} {data['total']:<10} {data['accuracy']:>7.2f}%")


def main():
    """Main experiment runner."""
    
    # Configuration
    NUM_QUESTIONS = None  # None = full dataset
    NOISE_SCALES = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    MAX_NEW_TOKENS = 100
    MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
    OUTPUT_DIR = "coconut_multi_gpu_results"
    
    run_distributed_experiment(
        num_questions=NUM_QUESTIONS,
        noise_scales=NOISE_SCALES,
        max_new_tokens=MAX_NEW_TOKENS,
        model_name=MODEL_NAME,
        output_dir=OUTPUT_DIR
    )


if __name__ == "__main__":
    main()