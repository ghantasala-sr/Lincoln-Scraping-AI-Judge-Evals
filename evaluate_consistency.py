import json
import os
from collections import defaultdict
from src.pipeline.judge import EventJudge

INPUT_FILE = "data/extracted/events_hybrid.json"
OUTPUT_FILE = "data/evaluated/consistency_report.json"

def load_events():
    if not os.path.exists(INPUT_FILE):
        print(f"Input file {INPUT_FILE} not found.")
        return []
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def is_lincoln_source(author: str, source_title: str) -> bool:
    """Heuristic to identify if the source is Lincoln or a primary proxy."""
    author_lower = author.lower()
    title_lower = source_title.lower()
    
    if "lincoln" in author_lower and "abraham" in author_lower:
        return True
    if "chew" in author_lower: # Robert S. Chew (proxy for Lincoln's orders)
        return True
    if "nicolay" in title_lower and "copy" in title_lower: # Nicolay Copy of Gettysburg Address
        return True
    if "abraham lincoln papers" in title_lower: # LoC Papers (Primary Source)
        return True
    # Check if title indicates it's a letter FROM Lincoln
    if "lincoln to" in title_lower:
        return True
        
    return False

def main():
    events = load_events()
    judge = EventJudge()
    
    # Group by event
    events_by_name = defaultdict(list)
    for e in events:
        events_by_name[e['event']].append(e)
        
    evaluation_results = []
    
    for event_name, event_list in events_by_name.items():
        print(f"\nEvaluating Event: {event_name}")
        
        # Split into Lincoln vs Others
        lincoln_sources = []
        other_sources = []
        
        for e in event_list:
            author = e.get('author', 'Unknown')
            title = e.get('source_title', '')
            if is_lincoln_source(author, title):
                lincoln_sources.append(e)
            else:
                other_sources.append(e)
                
        if not lincoln_sources:
            if "Ford's Theatre" in event_name:
                print(f"  - No Lincoln source found for {event_name}. (Historical Note: Lincoln could not write about his own assassination). Skipping.")
            else:
                print(f"  - No Lincoln source found for {event_name}. Skipping.")
            continue
            
        if not other_sources:
            print(f"  - No secondary sources found for {event_name}. Skipping.")
            continue
            
        # Compare each Lincoln source against each Other source
        # (Or just take the best Lincoln source? Let's do all pairs for thoroughness)
        
        for l_source in lincoln_sources:
            for o_source in other_sources:
                l_author = l_source.get('author', 'Lincoln')
                o_author = o_source.get('author', 'Other')
                
                print(f"  > Comparing {l_author} vs {o_author}...")
                
                result = judge.evaluate(event_name, l_source, o_source)
                
                if "error" not in result:
                    evaluation_entry = {
                        "event": event_name,
                        "source_a": {
                            "author": l_author,
                            "id": l_source.get('source_id')
                        },
                        "source_b": {
                            "author": o_author,
                            "id": o_source.get('source_id')
                        },
                        "evaluation": result
                    }
                    evaluation_results.append(evaluation_entry)
                    print(f"    Score: {result.get('consistency_score')}")
                else:
                    print("    Evaluation failed.")

    # Save results
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, indent=2)
        
    print(f"\nEvaluation complete. Saved {len(evaluation_results)} comparisons to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
