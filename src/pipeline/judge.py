import os
import time
import json
import re
import dspy
from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv

load_dotenv()

# ============ OUTPUT SCHEMA ============

class Contradiction(BaseModel):
    type: Literal["factual", "interpretive", "omission"]
    description: str
    quote_reference: str
    severity: Literal["minor", "moderate", "major"] = "moderate"

class ConsistencyReport(BaseModel):
    consistency_score: int = Field(ge=0, le=100)
    contradictions: List[Contradiction]
    reasoning: str
    confidence: Literal["high", "medium", "low"] = "medium"
    
    @validator('consistency_score')
    def validate_score(cls, v):
        if not 0 <= v <= 100:
            raise ValueError('Score must be between 0 and 100')
        return v

# ============ DSPY SIGNATURE ============

class CompareAccounts(dspy.Signature):
    """
    You are an impartial Historian and Logic Judge evaluating consistency between historical accounts.
    
    CRITICAL INSTRUCTIONS:
    1. NO OUTSIDE KNOWLEDGE: Base evaluation ONLY on provided text
    2. NO BIAS: Treat both sources as neutral observers
    3. EVIDENCE-BASED: Every contradiction must cite specific text
    4. PRECISION: Distinguish "direct contradiction" from "omission"
    
    CLASSIFICATION TYPES:
    - Factual: Direct contradiction on objective facts (dates, numbers, locations)
    - Interpretive: Difference in subjective analysis, motivation, or tone
    - Omission: One source includes significant details the other omits
    
    SEVERITY LEVELS:
    - Minor: Trivial difference, doesn't affect understanding
    - Moderate: Notable difference, worth mentioning
    - Major: Significant discrepancy, changes interpretation
    
    SCORING CALIBRATION:
    - 90-100: Near identical, only wording differs
    - 70-89: Same narrative, minor factual discrepancies
    - 50-69: Moderate differences, some omissions
    - 30-49: Significant contradictions
    - 0-29: Fundamentally conflicting accounts
    """
    
    event_name = dspy.InputField(desc="Name of the historical event")
    source_a_text = dspy.InputField(desc="Text/claims from Source A (primary source)")
    source_b_text = dspy.InputField(desc="Text/claims from Source B (secondary source)")
    
    consistency_report = dspy.OutputField(desc="""Return ONLY valid JSON. Do NOT add any text, notes, or attributions outside the JSON structure.
{
  "consistency_score": <0-100>,
  "contradictions": [
    {"type": "factual|interpretive|omission", "description": "...", "quote_reference": "Exact quote from text", "severity": "minor|moderate|major"}
  ],
  "reasoning": "Brief explanation of score",
  "confidence": "high|medium|low"
}""")

# ============ JUDGE CLASS ============

class EventJudge:
    def __init__(self, temperature: float = 0.0, model: str = None):
        """
        Initialize the DSPy-based judge.
        
        Args:
            temperature: LLM temperature (0 = deterministic, >0 = varied)
            model: Model name override
        """
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment")
        
        model_name = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        if "gemini" in model_name and "gemini/" not in model_name:
            model_name = f"gemini/{model_name}"
        
        self.lm = dspy.LM(
            model=model_name,
            api_key=api_key,
            temperature=temperature
        )
        dspy.settings.configure(lm=self.lm)
        
        self.judge = dspy.ChainOfThought(CompareAccounts)
        self.temperature = temperature
        self.model_name = model_name

    def _format_claims(self, source: Dict) -> str:
        """Format source claims into readable text."""
        claims = source.get('claims', [])
        author = source.get('author', 'Unknown')
        
        if not claims:
            return f"Author: {author}\nNo claims provided."
        
        claims_text = "\n".join([f"- {c}" for c in claims])
        return f"Author: {author}\nClaims:\n{claims_text}"

    def _clean_json(self, raw: str) -> str:
        """Clean LLM output to extract valid JSON."""
        # Remove markdown code blocks
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0]
        elif "```" in raw:
            parts = raw.split("```")
            if len(parts) >= 2:
                raw = parts[1]
        
        # Remove control characters
        raw = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', raw)
        
        return raw.strip()

    def _parse_response(self, raw_json: str) -> ConsistencyReport:
        """Parse and validate the JSON response."""
        cleaned = self._clean_json(raw_json)
        
        try:
            data = json.loads(cleaned, strict=False)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}\nRaw: {cleaned[:500]}")
        
        # Validate with Pydantic
        return ConsistencyReport(**data)

    def evaluate(self, event_name: str, source_a: Dict, source_b: Dict, 
                 max_retries: int = 3) -> Dict:
        """
        Compare two sources and evaluate consistency.
        
        Args:
            event_name: Name of historical event
            source_a: Dict with 'author' and 'claims' (primary source)
            source_b: Dict with 'author' and 'claims' (secondary source)
            max_retries: Number of retry attempts
            
        Returns:
            Dict with consistency_score, contradictions, reasoning
        """
        text_a = self._format_claims(source_a)
        text_b = self._format_claims(source_b)
        
        base_delay = 5
        last_error = None
        
        for attempt in range(max_retries):
            try:
                # Rate limit protection
                if attempt > 0:
                    time.sleep(base_delay * (2 ** attempt))
                
                # Call DSPy
                prediction = self.judge(
                    event_name=event_name,
                    source_a_text=text_a,
                    source_b_text=text_b
                )
                
                # Parse and validate
                report = self._parse_response(prediction.consistency_report)
                
                return {
                    "consistency_score": report.consistency_score,
                    "contradictions": [c.dict() for c in report.contradictions],
                    "reasoning": report.reasoning,
                    "confidence": report.confidence,
                    "metadata": {
                        "event": event_name,
                        "source_a_author": source_a.get('author'),
                        "source_b_author": source_b.get('author'),
                        "model": self.model_name,
                        "temperature": self.temperature
                    }
                }
                
            except Exception as e:
                last_error = e
                print(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                
                if "429" in str(e) or "rate" in str(e).lower():
                    continue  # Retry on rate limit
                elif attempt == max_retries - 1:
                    break  # Last attempt, give up
        
        return {
            "error": f"Failed after {max_retries} attempts: {last_error}",
            "consistency_score": None,
            "contradictions": [],
            "reasoning": None
        }

# ============ STATISTICAL VALIDATION ============

class JudgeValidator:
    """Run statistical validation experiments on the judge."""
    
    def __init__(self, judge_class=EventJudge):
        self.judge_class = judge_class
    
    def run_consistency_test(self, event_name: str, source_a: Dict, 
                             source_b: Dict, n_runs: int = 5, 
                             temperature: float = 0.7) -> Dict:
        """
        Test self-consistency by running same comparison multiple times.
        Returns variance statistics.
        """
        judge = self.judge_class(temperature=temperature)
        scores = []
        
        for i in range(n_runs):
            print(f"  Consistency run {i+1}/{n_runs}")
            # Inject random noise to force fresh generation (bypass ANY cache)
            import random
            noise = f" [Run ID: {random.randint(1000, 9999)}]"
            
            # We can't easily modify the input text without changing the signature, 
            # but we can append it to the event name which is an input field.
            result = judge.evaluate(event_name + noise, source_a, source_b)
            
            if result.get("consistency_score") is not None:
                scores.append(result["consistency_score"])
            
            time.sleep(2)  # Avoid rate limits
        
        if not scores:
            return {"error": "No successful runs"}
        
        import numpy as np
        return {
            "scores": scores,
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "variance": float(np.var(scores)),
            "min": min(scores),
            "max": max(scores),
            "range": max(scores) - min(scores),
            "n_runs": len(scores),
            "temperature": temperature
        }
    
    def run_kappa_test(self, test_pairs: List[Dict], human_labels: List[int],
                       threshold: int = 50) -> Dict:
        """
        Calculate Cohen's Kappa against human labels.
        
        Args:
            test_pairs: List of {"event", "source_a", "source_b"} dicts
            human_labels: List of 0 (contradictory) or 1 (consistent)
            threshold: Score above which LLM is "consistent"
        """
        from sklearn.metrics import cohen_kappa_score, confusion_matrix
        
        judge = self.judge_class(temperature=0)  # Deterministic
        llm_scores = []
        
        for i, pair in enumerate(test_pairs):
            print(f"  Kappa test {i+1}/{len(test_pairs)}")
            
            # Inject noise to bypass cache (even for Kappa)
            import random
            noise = f" [Run ID: {random.randint(1000, 9999)}]"
            
            result = judge.evaluate(
                pair["event"] + noise, 
                pair["source_a"], 
                pair["source_b"]
            )
            llm_scores.append(result.get("consistency_score", 0))
            time.sleep(2)
        
        # Convert to binary
        llm_labels = [1 if s >= threshold else 0 for s in llm_scores]
        
        # Calculate Kappa
        kappa = cohen_kappa_score(human_labels, llm_labels)
        cm = confusion_matrix(human_labels, llm_labels)
        agreement = sum(h == l for h, l in zip(human_labels, llm_labels)) / len(human_labels)
        
        return {
            "kappa": float(kappa),
            "raw_agreement": float(agreement),
            "confusion_matrix": cm.tolist(),
            "llm_scores": llm_scores,
            "llm_labels": llm_labels,
            "human_labels": human_labels,
            "threshold": threshold,
            "interpretation": self._interpret_kappa(kappa)
        }
    
    def _interpret_kappa(self, kappa: float) -> str:
        if kappa < 0:
            return "Less than chance - check for errors"
        elif kappa < 0.21:
            return "Slight agreement"
        elif kappa < 0.41:
            return "Fair agreement"
        elif kappa < 0.61:
            return "Moderate agreement"
        elif kappa < 0.81:
            return "Substantial agreement"
        else:
            return "Almost perfect agreement"



# ============ USAGE EXAMPLE ============

if __name__ == "__main__":
    # Example usage
    judge = EventJudge(temperature=0)
    
    source_lincoln = {
        "author": "Abraham Lincoln",
        "claims": [
            "I waited for the South to fire first",
            "The decision to resupply was mine alone",
            "We sent provisions only, not arms"
        ]
    }
    
    source_biographer = {
        "author": "Doris Kearns Goodwin",
        "claims": [
            "Lincoln deliberately provoked the confrontation",
            "The cabinet was divided on the decision",
            "The resupply mission included both food and ammunition"
        ]
    }
    
    result = judge.evaluate("Fort Sumter Decision", source_lincoln, source_biographer)
    print(json.dumps(result, indent=2))
