import json
import os
import time
import random
import numpy as np
import google.generativeai as genai
from typing import List, Dict
from dotenv import load_dotenv

# Load environment variables
# load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

genai.configure(api_key=api_key)

# Configuration
MODEL_NAME = "gemini-2.5-flash"
TEMPERATURE = 0.7

def load_manual_data():
    with open("data/manual_labels.json", "r") as f:
        return json.load(f)

def get_base_prompt(event_name, source_a, source_b):
    return f"""You are an expert historian evaluating the consistency between two historical accounts.

EVENT: {event_name}

SOURCE A (Primary):
{json.dumps(source_a, indent=2)}

SOURCE B (Secondary):
{json.dumps(source_b, indent=2)}

TASK:
Compare Source B against Source A. Identify any contradictions, omissions, or factual errors in Source B relative to Source A.
"""

def get_json_instruction():
    return """
Return ONLY valid JSON in the following format. Do not include markdown formatting (```json ... ```).
{
  "consistency_score": <0-100>,
  "contradictions": [
    {"type": "factual|interpretive|omission", "description": "...", "quote_reference": "...", "severity": "minor|moderate|major"}
  ],
  "reasoning": "Brief explanation",
  "confidence": "high|medium|low"
}
"""

def build_zeroshot_prompt(event_name, source_a, source_b):
    return f"""{get_base_prompt(event_name, source_a, source_b)}

{get_json_instruction()}
"""

def build_cot_prompt(event_name, source_a, source_b):
    return f"""{get_base_prompt(event_name, source_a, source_b)}

Let's think step by step.
1. Analyze the key claims in Source A.
2. Analyze the key claims in Source B.
3. Compare them to find discrepancies.
4. Determine the severity of any contradictions.
5. Assign a consistency score.

{get_json_instruction()}
"""

def build_fewshot_prompt(event_name, source_a, source_b, examples):
    prompt = "Here are examples of how to evaluate historical consistency:\n\n"
    
    for ex in examples:
        prompt += f"--- EXAMPLE ---\n"
        prompt += get_base_prompt(ex['event'], ex['source_a'], ex['source_b'])
        prompt += "\nRESPONSE:\n"
        # Construct a sample response based on the label
        score = 85 if ex['human_label'] == 1 else 10
        reasoning = "Consistent accounts." if ex['human_label'] == 1 else "Major contradiction found."
        response = {
            "consistency_score": score,
            "contradictions": [],
            "reasoning": reasoning,
            "confidence": "high"
        }
        prompt += json.dumps(response, indent=2)
        prompt += "\n\n"
        
    prompt += "--- NEW TASK ---\n"
    prompt += get_base_prompt(event_name, source_a, source_b)
    prompt += get_json_instruction()
    return prompt

def call_llm(prompt):
    model = genai.GenerativeModel(MODEL_NAME)
    try:
        # Inject noise to bypass cache
        noise = f"\n[System Note: Run ID {random.randint(1000, 9999)}]"
        response = model.generate_content(
            prompt + noise,
            generation_config=genai.types.GenerationConfig(
                temperature=TEMPERATURE,
                response_mime_type="application/json"
            )
        )
        return response.text
    except Exception as e:
        print(f"API Error: {e}")
        return None

def parse_response(text):
    try:
        # Clean markdown if present
        text = text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except json.JSONDecodeError:
        return None

def run_test(strategy_name, prompt_builder, pairs, examples=None, n_runs=5):
    print(f"\n=== Testing Strategy: {strategy_name.upper()} ===")
    all_stds = []
    details = []
    
    for i, pair in enumerate(pairs):
        print(f"  Pair {i+1}: {pair['event']}")
        scores = []
        for r in range(n_runs):
            if strategy_name == "fewshot":
                prompt = prompt_builder(pair['event'], pair['source_a'], pair['source_b'], examples)
            else:
                prompt = prompt_builder(pair['event'], pair['source_a'], pair['source_b'])
                
            response_text = call_llm(prompt)
            if response_text:
                data = parse_response(response_text)
                if data and "consistency_score" in data:
                    scores.append(data["consistency_score"])
            
            time.sleep(1) # Rate limit
            
        if scores:
            std = np.std(scores)
            all_stds.append(std)
            details.append({
                "event": pair['event'],
                "scores": scores,
                "std_dev": std,
                "mean": np.mean(scores)
            })
            print(f"    Scores: {scores} -> Std: {std:.2f}")
        else:
            print("    No valid scores obtained.")
            
    avg_std = np.mean(all_stds) if all_stds else 0.0
    return avg_std, details

def main():
    print("Loading data...")
    data = load_manual_data()
    # Use indices 1, 2, 3 for testing (avoiding the examples at 0 and -1)
    test_pairs = data[1:4]
    examples = [data[0], data[-1]] # One consistent, one contradictory
    
    results = {}
    all_details = {}
    
    # 1. Zero-Shot
    std, details = run_test("zeroshot", build_zeroshot_prompt, test_pairs)
    results["Zero-Shot"] = std
    all_details["Zero-Shot"] = details
    
    # 2. Chain-of-Thought
    std, details = run_test("cot", build_cot_prompt, test_pairs)
    results["Chain-of-Thought"] = std
    all_details["Chain-of-Thought"] = details
    
    # 3. Few-Shot
    std, details = run_test("fewshot", build_fewshot_prompt, test_pairs, examples=examples)
    results["Few-Shot"] = std
    all_details["Few-Shot"] = details
    
    print("\n" + "="*30)
    print("FINAL COMPARISON RESULTS (Std Dev - Lower is Better)")
    print("="*30)
    best_strat = min(results, key=results.get)
    for s, std in results.items():
        print(f"{s.ljust(20)}: {std:.4f}")
        
    print(f"\nWinner: {best_strat} is the most stable strategy.")
    
    # Save to JSON
    output_path = "data/evaluated/prompt_comparison_report.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "summary": results,
            "winner": best_strat,
            "details": all_details
        }, f, indent=2)
    print(f"\nFull report saved to {output_path}")

if __name__ == "__main__":
    main()
