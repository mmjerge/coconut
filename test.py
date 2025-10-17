"""
Test Coconut model with varying noise levels on GSM8K using Llama 3.1 8B Instruct.
Tests different noise scales to see when the system breaks and outputs become nonsensical.
Results are saved to JSON for analysis.
"""

import torch
import json
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from coconut import Coconut
from datasets import load_dataset
from typing import List, Dict, Any
import re

def setup_model_and_tokenizer(model_name: str = "Qwen/Qwen3-0.6B"):
    """Initialize Llama 3.1 8B Instruct model and add special tokens."""
    print(f"Loading {model_name}...")
    print("Note: You may need to authenticate with Hugging Face and accept the Llama license.")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=True
    )
    
    # Load model
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # Use FP16 for efficiency
        device_map="cpu",  # Automatically handle device placement
        trust_remote_code=True
    )
    
    # Add special tokens
    special_tokens = {
        'additional_special_tokens': ['<|latent|>', '<|start-latent|>', '<|end-latent|>']
    }
    num_added = tokenizer.add_special_tokens(special_tokens)
    print(f"Added {num_added} special tokens")
    
    # Resize model embeddings to accommodate new tokens
    base_model.resize_token_embeddings(len(tokenizer))
    
    # Get token IDs
    latent_token_id = tokenizer.convert_tokens_to_ids('<|latent|>')
    start_latent_id = tokenizer.convert_tokens_to_ids('<|start-latent|>')
    end_latent_id = tokenizer.convert_tokens_to_ids('<|end-latent|>')
    eos_token_id = tokenizer.eos_token_id
    
    # Set padding token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Model loaded successfully!")
    print(f"Special token IDs:")
    print(f"  <|latent|>: {latent_token_id}")
    print(f"  <|start-latent|>: {start_latent_id}")
    print(f"  <|end-latent|>: {end_latent_id}")
    print(f"  EOS: {eos_token_id}")
    
    return tokenizer, base_model, latent_token_id, start_latent_id, end_latent_id, eos_token_id


def load_gsm8k_questions(num_questions: int = 5) -> List[Dict[str, str]]:
    """Load first N questions from GSM8K dataset."""
    print(f"\nLoading first {num_questions} questions from GSM8K...")
    
    try:
        dataset = load_dataset("gsm8k", "main", split="test")
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
    
    except Exception as e:
        print(f"Error loading GSM8K: {e}")
        print("Using fallback example questions...")
        
        return [
            {
                "question": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
                "answer": "#### 18",
                "question_id": 0
            },
            {
                "question": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
                "answer": "#### 3",
                "question_id": 1
            },
            {
                "question": "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?",
                "answer": "#### 70000",
                "question_id": 2
            },
            {
                "question": "James decides to run 3 sprints 3 times a week. He runs 60 meters each sprint. How many total meters does he run a week?",
                "answer": "#### 540",
                "question_id": 3
            },
            {
                "question": "Every day, Wendi feeds each of her chickens three cups of mixed chicken feed. She gives her flock of chickens 15 cups in the morning and 25 cups in the afternoon. How many cups does she need in the final meal if she has 20 chickens that each eat 3 cups per day?",
                "answer": "#### 20",
                "question_id": 4
            }
        ]


def extract_numerical_answer(text: str) -> str:
    """Extract numerical answer from generated text."""
    patterns = [
        r'####\s*(\d+)',  # GSM8K format
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
    
    # Return last number found
    numbers = re.findall(r'\d+', text)
    if numbers:
        return numbers[-1]
    
    return "NO_ANSWER_FOUND"


def create_coconut_input(tokenizer, question: str, start_latent_id: int, end_latent_id: int):
    """Create input for Coconut model in TRUE METHOD format with Llama-style prompting."""
    # Use Llama 3.1 Instruct format
    prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{question}

Please solve this step by step and provide the final numerical answer.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    
    # Tokenize
    question_ids = tokenizer.encode(prompt, return_tensors='pt', add_special_tokens=False)
    
    # Add latent markers for continuous thought
    start_token = torch.tensor([[start_latent_id]])
    end_token = torch.tensor([[end_latent_id]])
    
    input_ids = torch.cat([question_ids, start_token, end_token], dim=1)
    
    return input_ids


def extract_generated_only(tokenizer, full_ids: torch.Tensor, original_input_ids: torch.Tensor, end_latent_id: int) -> str:
    """Extract only the newly generated tokens after <end-latent>."""
    end_latent_positions = (original_input_ids[0] == end_latent_id).nonzero(as_tuple=True)[0]
    
    if len(end_latent_positions) > 0:
        # Account for 8 virtual latent tokens inserted during TRUE METHOD
        end_latent_pos_expanded = end_latent_positions[0].item() + 8
        
        if full_ids.shape[1] > end_latent_pos_expanded + 1:
            generated_ids = full_ids[0, end_latent_pos_expanded + 1:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            return generated_text.strip()
    
    # Fallback
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
    
    # Create input
    input_ids = create_coconut_input(tokenizer, question, start_latent_id, end_latent_id)
    original_input_ids = input_ids.clone()
    input_ids = input_ids.to(device)
    
    # Generate response WITH NOISE
    with torch.no_grad():
        try:
            generated_ids = coconut_model.generate(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=max_new_tokens,
                noise_scale=noise_scale
            )
            
            # Extract ONLY the generated portion
            generated_text = extract_generated_only(
                tokenizer, 
                generated_ids, 
                original_input_ids, 
                end_latent_id
            )
            
            full_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            
            return {
                "success": True,
                "generated_text": generated_text,
                "full_text": full_text,
                "num_tokens": generated_ids.shape[1],
                "num_new_tokens": generated_ids.shape[1] - original_input_ids.shape[1],
                "error": None
            }
            
        except Exception as e:
            return {
                "success": False,
                "generated_text": "",
                "full_text": "",
                "num_tokens": 0,
                "num_new_tokens": 0,
                "error": str(e)
            }


def run_noise_experiment(
    num_questions: int = 5,
    noise_scales: List[float] = None,
    max_new_tokens: int = 100,
    model_name: str = "Qwen/Qwen3-0.6B"
) -> Dict[str, Any]:
    """Run complete noise experiment on GSM8K questions."""
    
    if noise_scales is None:
        noise_scales = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("="*70)
    print("COCONUT NOISE ROBUSTNESS EXPERIMENT - LLAMA 3.1 8B INSTRUCT")
    print("="*70)
    print(f"Testing {len(noise_scales)} noise scales on {num_questions} GSM8K questions")
    print(f"Noise scales: {noise_scales}")
    print(f"Device: {device}")
    print(f"Max new tokens: {max_new_tokens}")
    
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
    questions = load_gsm8k_questions(num_questions)
    
    # Run experiments
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
        "questions": []
    }
    
    for q_idx, q_data in enumerate(questions):
        print(f"\n{'='*70}")
        print(f"Question {q_idx + 1}/{len(questions)}")
        print(f"{'='*70}")
        print(f"Q: {q_data['question'][:100]}...")
        print(f"Expected Answer: {q_data['answer']}")
        
        question_results = {
            "question_id": q_data["question_id"],
            "question": q_data["question"],
            "expected_answer": q_data["answer"],
            "noise_tests": []
        }
        
        for noise_scale in noise_scales:
            print(f"\n  Testing noise={noise_scale}...", end=" ")
            
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
                
                print(f"✓ Answer: {generated_answer} | Expected: {expected_answer} | {'✓' if is_correct else '✗'}")
                
                question_results["noise_tests"].append({
                    "noise_scale": noise_scale,
                    "generated_text": result["generated_text"],
                    "generated_answer": generated_answer,
                    "expected_answer": expected_answer,
                    "is_correct": is_correct,
                    "num_tokens": result["num_tokens"],
                    "num_new_tokens": result["num_new_tokens"],
                    "success": True,
                    "error": None
                })
            else:
                print(f"✗ Error: {result['error']}")
                
                question_results["noise_tests"].append({
                    "noise_scale": noise_scale,
                    "generated_text": "",
                    "generated_answer": None,
                    "expected_answer": extract_numerical_answer(q_data["answer"]),
                    "is_correct": False,
                    "num_tokens": 0,
                    "num_new_tokens": 0,
                    "success": False,
                    "error": result["error"]
                })
        
        results["questions"].append(question_results)
    
    return results


def analyze_results(results: Dict[str, Any]) -> None:
    """Print analysis of the results."""
    print("\n" + "="*70)
    print("RESULTS ANALYSIS")
    print("="*70)
    
    noise_scales = results["experiment_info"]["noise_scales"]
    
    print("\nAccuracy by Noise Scale:")
    print("-" * 70)
    print(f"{'Noise':<8} {'Correct':<8} {'Total':<8} {'Accuracy':<10} {'Status'}")
    print("-" * 70)
    
    for noise_scale in noise_scales:
        correct = 0
        total = 0
        
        for question in results["questions"]:
            for test in question["noise_tests"]:
                if test["noise_scale"] == noise_scale and test["success"]:
                    total += 1
                    if test["is_correct"]:
                        correct += 1
        
        accuracy = (correct / total * 100) if total > 0 else 0
        
        if accuracy >= 60:
            status = "✓ Good"
        elif accuracy >= 20:
            status = "⚠ Degraded"
        else:
            status = "✗ Broken"
        
        print(f"{noise_scale:<8.2f} {correct:<8} {total:<8} {accuracy:>6.1f}%     {status}")
    
    # Show example outputs
    print("\n" + "="*70)
    print("SAMPLE OUTPUTS (First Question)")
    print("="*70)
    
    if results["questions"]:
        first_q = results["questions"][0]
        print(f"\nQuestion: {first_q['question'][:120]}...")
        print(f"Expected: {first_q['expected_answer']}")
        print("\n" + "-" * 70)
        
        for test in first_q["noise_tests"]:
            noise = test["noise_scale"]
            answer = test["generated_answer"]
            correct = "✓" if test["is_correct"] else "✗"
            output = test["generated_text"][:100]
            
            print(f"\nNoise={noise:>5.2f}: {correct} [{answer}]")
            print(f"  {output}...")


def save_results(results: Dict[str, Any], filename: str = "coconut_llama_noise_results.json") -> None:
    """Save results to JSON file."""
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {filename}")


def main():
    """Main experiment runner."""
    
    # Configuration
    NUM_QUESTIONS = 5
    NOISE_SCALES = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    MAX_NEW_TOKENS = 100
    MODEL_NAME = "Qwen/Qwen3-0.6B"
    OUTPUT_FILE = "coconut_llama_noise_results.json"
    
    # Run experiment
    results = run_noise_experiment(
        num_questions=NUM_QUESTIONS,
        noise_scales=NOISE_SCALES,
        max_new_tokens=MAX_NEW_TOKENS,
        model_name=MODEL_NAME
    )
    
    # Analyze
    analyze_results(results)
    
    # Save
    save_results(results, OUTPUT_FILE)
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()