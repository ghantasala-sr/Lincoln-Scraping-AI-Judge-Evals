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
        You are an expert historian analyzing historical documents.
        
        Task: Extract information specifically about the event "{event_name}" from the provided text.
        
        Text Source Author: {author}
        
        Context:
        {context}
        
        Instructions:
        1. Identify specific factual claims made about the event in this text.
        2. Extract any temporal details (dates, times) mentioned in relation to the event.
        3. Analyze the tone of the text regarding this event (e.g., sympathetic, critical, neutral, urgent).
        4. Output strictly valid JSON.
        
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
        
        If the text does not contain relevant information about "{event_name}", return null.
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
