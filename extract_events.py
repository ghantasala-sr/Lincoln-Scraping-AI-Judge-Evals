import json
import os
from typing import List, Dict
from src.pipeline.retriever import ContextRetriever
from src.pipeline.extractor import EventExtractor
from src.models.schema import Document

# Constants
DATA_DIR = "data/normalized"
OUTPUT_DIR = "data/extracted"
EVENTS = [
    "Election Night 1860",
    "Fort Sumter Decision",
    "Gettysburg Address",
    "Second Inaugural Address",
    "Ford's Theatre Assassination"
]

def load_documents() -> List[Dict]:
    """Load all documents from normalized data directory."""
    documents = []
    
    # Load LoC data
    loc_path = os.path.join(DATA_DIR, "loc_dataset.json")
    if os.path.exists(loc_path):
        with open(loc_path, 'r', encoding='utf-8') as f:
            documents.extend(json.load(f))
            
    # Load Gutenberg data
    gutenberg_path = os.path.join(DATA_DIR, "gutenberg_dataset.json")
    if os.path.exists(gutenberg_path):
        with open(gutenberg_path, 'r', encoding='utf-8') as f:
            documents.extend(json.load(f))
            
    return documents

def main():
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Initialize pipeline components
    retriever = ContextRetriever(chunk_size=2000, overlap=200)
    extractor = EventExtractor()
    
    documents = load_documents()
    print(f"Loaded {len(documents)} documents.")
    
    all_extractions = []
    
    for doc in documents:
        doc_id = doc.get('id')
        title = doc.get('title', 'Unknown')
        author = doc.get('from', 'Unknown') # 'from_' field in schema mapped to 'from' in JSON? Check schema.
        # Schema has 'from_', but JSON dump uses alias 'from' if configured.
        # Let's check raw JSON to be sure.
        if 'from' in doc:
            author = doc['from']
        elif 'from_' in doc:
            author = doc['from_']
            
        content = doc.get('content', '')
        
        if not content:
            print(f"Skipping {doc_id} (no content)")
            continue
            
        print(f"Processing {doc_id}: {title[:50]}...")
        
        for event in EVENTS:
            # 1. Retrieve relevant chunks
            chunks = retriever.retrieve(content, event, top_k=3)
            
            if not chunks:
                # print(f"  - No relevant text found for {event}")
                continue
                
            print(f"  + Found relevant text for {event}. Extracting...")
            
            # 2. Extract information
            extraction = extractor.extract(event, chunks, author)
            
            if extraction:
                # Add metadata to extraction
                extraction['source_id'] = doc_id
                extraction['source_title'] = title
                all_extractions.append(extraction)
                print(f"  > Extracted info for {event}")
            else:
                print(f"  - LLM found no relevant info for {event}")

    # Save results
    output_path = os.path.join(OUTPUT_DIR, "events.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_extractions, f, indent=2, ensure_ascii=False)
        
    print(f"\nExtraction complete. Saved {len(all_extractions)} events to {output_path}")

if __name__ == "__main__":
    main()
