import os
import json
import google.generativeai as genai
from typing import List, Dict, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class EventExtractor:
    def __init__(self, model_name: str = None):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        if not model_name:
            model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
            
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def extract(self, event_name: str, chunks: List[str], author: str = "Unknown") -> Optional[Dict]:
        """
        Extracts event information from the provided text chunks using Gemini.
        """
        if not chunks:
            return None

        context = "\n\n".join(chunks)
        
        prompt = f"""
        You are an expert historian specializing in 19th-century American history and primary source analysis.

    ## TASK
    Extract all information about the event "{event_name}" from the provided historical text.

    ## SOURCE INFORMATION
    - **Author**: {author}

    ## TEXT TO ANALYZE
    {context}

    ## EXTRACTION GUIDELINES

    ### What Constitutes a "Claim"
    A claim is a specific, verifiable statement of fact. Extract claims that are:
    - **Factual assertions**: Names, dates, locations, actions, sequences of events
    - **Attributed statements**: What people said or wrote
    - **Quantitative details**: Numbers, durations, counts

    Do NOT extract:
    - General commentary or opinions without factual basis
    - Information unrelated to "{event_name}"
    - Speculative or hypothetical statements (unless clearly attributed)

    ### Claim Quality Standards
    - Each claim should be a single, atomic fact
    - Claims should be directly supported by the text
    - Preserve specificity (don't generalize "many people" if text says "500 soldiers")

    ## EXAMPLES

    ### Good Claims (Specific, Factual):
    - "Lincoln received 180 electoral votes in the 1860 election"
    - "The notification was delivered to Governor Pickens on April 8, 1861"
    - "Major Anderson commanded a garrison of 76 combatants at Fort Sumter"

    ### Bad Claims (Vague, Opinion-based):
    - "Lincoln was a great president" (opinion)
    - "The election was important" (vague)
    - "Many people supported Lincoln" (unspecific)
        
        Output Format:
        {{
            "event": "{event_name}",
            "author": "{author}",
            "claims": [
                "claim 1",
                "claim 2"
            ],
            "temporal_details": {{
                "date": "YYYY-MM-DD or description",
                "time": "HH:MM or description"
            }},
            "tone": "Description of tone"
        }}
        
        ## CRITICAL RULES
        1. **Only extract what is explicitly stated** - Do not infer or add information
        2. **Preserve original precision** - If text says "about 500", don't say "500"
        3. **Cite evidence for tone** - Explain why you chose that tone classification
        4. **Return null if no relevant information** - Better to return null than hallucinate

        If the text contains NO information about "{event_name}", return exactly: null
        
        """
        
        import time
        
        max_retries = 5
        base_delay = 2
        
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config={"response_mime_type": "application/json"}
                )
                
                if not response.text:
                    return None
                    
                result = json.loads(response.text)
                time.sleep(10) # Enforce 10s gap between calls
                return result
                
            except Exception as e:
                if "429" in str(e) or "Quota exceeded" in str(e):
                    delay = base_delay * (2 ** attempt)
                    print(f"Rate limit hit for {event_name}. Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    print(f"Error extracting event {event_name}: {e}")
                    return None
        
        print(f"Max retries exceeded for {event_name}")
        time.sleep(10) # Enforce 10s gap even on failure
        return None
