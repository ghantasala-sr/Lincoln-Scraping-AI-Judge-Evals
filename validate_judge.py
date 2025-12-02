import json
import os
import dspy
from typing import List, Dict
from src.pipeline.judge import EventJudge, JudgeValidator

# Disable DSPy Cache to ensure statistical validity
os.environ["DSP_CACHEBOOL"] = "False"

def load_events(filepath: str) -> List[Dict]:
    with open(filepath, 'r') as f:
        return json.load(f)

def get_comparison_pairs(events: List[Dict], limit: int = 10) -> List[Dict]:
    """
    Extracts pairs of (Lincoln Source, Other Source) for validation.
    """
    # Group by event name
    grouped_events = {}
    for e in events:
        name = e['event']
        if name not in grouped_events:
            grouped_events[name] = []
        grouped_events[name].append(e)
        
    pairs = []
    for event_name, sources in grouped_events.items():
        # Identify Lincoln source
        lincoln_source = None
        other_sources = []
        
        for source in sources:
            # Simple heuristic: check author or title
            author = source.get('author', '').lower()
            title = source.get('source_title', '').lower() # Changed from 'title' to 'source_title' based on JSON
            
            is_lincoln = (
                ("lincoln" in author and "abraham" in author) or
                "chew" in author or # Robert S. Chew
                ("nicolay" in title and "copy" in title) or
                "abraham lincoln papers" in title or
                "lincoln to" in title
            )
            
            if is_lincoln:
                lincoln_source = source
            else:
                other_sources.append(source)
        
        if lincoln_source and other_sources:
            for other in other_sources:
                pairs.append({
                    "event": event_name,
                    "source_a": lincoln_source,
                    "source_b": other
                })
                if len(pairs) >= limit:
                    return pairs
    return pairs

def main():
    print("=== STARTING VALIDATION SUITE ===")
    
    # ---------------------------------------------------------
    # 1. Inter-Rater Agreement (Cohen's Kappa)
    # ---------------------------------------------------------
    print("\n[1/2] Loading manually labeled data for Kappa Test...")
    try:
        with open("data/manual_labels.json", "r") as f:
            manual_pairs = json.load(f)
            
        # Extract human labels
        human_labels = [p['human_label'] for p in manual_pairs]
        
        if any(l is None for l in human_labels):
            print("Error: Some pairs are missing 'human_label'. Skipping Kappa test.")
        else:
            print(f"Loaded {len(manual_pairs)} labeled pairs.")
            
            validator = JudgeValidator(EventJudge)
            print(f"Running Kappa Test on {len(manual_pairs)} pairs...")
            
            kappa_results = validator.run_kappa_test(manual_pairs, human_labels, threshold=50)
            
            print("\n" + "-"*30)
            print("KAPPA TEST RESULTS")
            print("-"*30)
            print(json.dumps(kappa_results, indent=2))
            
            os.makedirs("data/evaluated", exist_ok=True)
            with open("data/evaluated/kappa_report.json", "w") as f:
                json.dump(kappa_results, f, indent=2)
            print("Saved to data/evaluated/kappa_report.json")
            
    except FileNotFoundError:
        print("Error: data/manual_labels.json not found. Skipping Kappa test.")

    # ---------------------------------------------------------
    # 2. Full Dataset Self-Consistency
    # ---------------------------------------------------------
    print("\n[2/2] Loading full dataset for Self-Consistency Test...")
    try:
        events = load_events("data/extracted/events_hybrid.json")
        all_pairs = get_comparison_pairs(events, limit=1000)
        print(f"Found {len(all_pairs)} comparison pairs in the full dataset.")
        
        validator = JudgeValidator(EventJudge)
        print(f"Running Self-Consistency Test on ALL {len(all_pairs)} pairs (5 runs each)...")
        
        pair_stats = []
        for i, pair in enumerate(all_pairs):
            print(f"Processing {i+1}/{len(all_pairs)}: {pair['event']} ({pair['source_b']['author']})")
            stats = validator.run_consistency_test(
                pair['event'], 
                pair['source_a'], 
                pair['source_b'], 
                n_runs=5, 
                temperature=0.7
            )
            
            if "error" not in stats:
                pair_stats.append({
                    "event": pair['event'],
                    "source_b": pair['source_b'].get('author', 'Unknown'),
                    "mean_score": stats['mean'],
                    "std_dev": stats['std'],
                    "scores": stats['scores']
                })
        
        if pair_stats:
            avg_std = sum(p['std_dev'] for p in pair_stats) / len(pair_stats)
            avg_mean = sum(p['mean_score'] for p in pair_stats) / len(pair_stats)
            
            print("\n" + "-"*30)
            print("FULL CONSISTENCY RESULTS")
            print("-"*30)
            print(f"Total Pairs: {len(pair_stats)}")
            print(f"Avg Std Dev: {avg_std:.4f}")
            print(f"Avg Score:   {avg_mean:.4f}")
            
            with open("data/evaluated/full_consistency_report.json", "w") as f:
                json.dump(pair_stats, f, indent=2)
            print("Saved to data/evaluated/full_consistency_report.json")
            
    except FileNotFoundError:
        print("Error: data/extracted/events_hybrid.json not found. Skipping Consistency test.")

    print("\n=== VALIDATION SUITE COMPLETE ===")

if __name__ == "__main__":
    main()
