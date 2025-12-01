import os
import time
import numpy as np
import google.generativeai as genai
from typing import List, Dict, Tuple
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class HybridRetriever:
    def __init__(self, chunk_size: int = 2000, overlap: int = 200, alpha: float = 0.7):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.alpha = alpha  # Weight for vector score (1-alpha for keyword score)
        
        # Configure Gemini
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        genai.configure(api_key=api_key)
        self.embedding_model = "models/text-embedding-004"
        
        # Keywords for the 5 key events (same as ContextRetriever)
        self.event_keywords = {
            "Election Night 1860": [
                "election", "1860", "November", "Lincoln", "Douglas", "Breckinridge", "Bell",
                "returns", "telegraph", "Springfield", "victory", "president"
            ],
            "Fort Sumter Decision": [
                "Sumter", "Anderson", "Seward", "cabinet", "provision", "resupply", 
                "Beauregard", "Charleston", "April", "1861", "attack", "surrender"
            ],
            "Gettysburg Address": [
                "Gettysburg", "cemetery", "dedication", "November", "1863", "score", 
                "conceived", "proposition", "Everett", "speech", "address"
            ],
            "Second Inaugural Address": [
                "inaugural", "second", "March", "1865", "malice", "charity", "God", 
                "scourge", "war", "address", "speech"
            ],
            "Ford's Theatre Assassination": [
                "Ford", "theatre", "theater", "assassination", "Booth", "shot", "pistol",
                "April", "1865", "Good Friday", "Laura Keene", "box", "president"
            ]
        }
        
        # Cache for event query embeddings
        self.query_embeddings = {}

    def chunk_text(self, text: str) -> List[str]:
        """Splits text into overlapping chunks of words."""
        if not text:
            return []
            
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunk = " ".join(words[i : i + self.chunk_size])
            chunks.append(chunk)
            
        return chunks

    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a single text string."""
        # Add delay to avoid rate limits
        time.sleep(1) 
        try:
            result = genai.embed_content(
                model=self.embedding_model,
                content=text,
                task_type="retrieval_document",
                title="Historical Text Chunk"
            )
            return np.array(result['embedding'])
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return np.zeros(768) # Return zero vector on failure

    def get_query_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a query string."""
        if text in self.query_embeddings:
            return self.query_embeddings[text]
            
        time.sleep(1)
        try:
            result = genai.embed_content(
                model=self.embedding_model,
                content=text,
                task_type="retrieval_query"
            )
            embedding = np.array(result['embedding'])
            self.query_embeddings[text] = embedding
            return embedding
        except Exception as e:
            print(f"Error getting query embedding: {e}")
            return np.zeros(768)

    def cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return np.dot(v1, v2) / (norm1 * norm2)

    def retrieve(self, text: str, event_name: str, top_k: int = 3) -> List[str]:
        """
        Retrieves the most relevant chunks using a hybrid score.
        Hybrid Score = alpha * Vector_Score + (1 - alpha) * Keyword_Score
        """
        if not text:
            return []
            
        chunks = self.chunk_text(text)
        if not chunks:
            return []
            
        # 1. Calculate Keyword Scores
        keywords = self.event_keywords.get(event_name, [])
        keyword_scores = []
        for chunk in chunks:
            score = 0
            chunk_lower = chunk.lower()
            for keyword in keywords:
                if keyword.lower() in chunk_lower:
                    score += 1
            keyword_scores.append(score)
            
        # Normalize keyword scores
        max_kw_score = max(keyword_scores) if keyword_scores else 1
        if max_kw_score == 0: max_kw_score = 1
        normalized_kw_scores = [s / max_kw_score for s in keyword_scores]
        
        # Optimization: Only embed chunks with non-zero keyword score?
        # Or embed all? Embedding all is expensive for books.
        # Hybrid strategy often filters first.
        # Let's filter to top 10 chunks by keyword score, then re-rank with vectors.
        # This saves API calls.
        
        chunk_indices = [i for i, score in enumerate(keyword_scores) if score > 0]
        if not chunk_indices:
            return []
            
        # Sort by keyword score and take top 10 candidates
        chunk_indices.sort(key=lambda i: keyword_scores[i], reverse=True)
        candidate_indices = chunk_indices[:10]
        
        # 2. Calculate Vector Scores for candidates
        query_embedding = self.get_query_embedding(f"Details about {event_name}")
        
        hybrid_scores = []
        for i in candidate_indices:
            chunk = chunks[i]
            chunk_embedding = self.get_embedding(chunk)
            vector_score = self.cosine_similarity(query_embedding, chunk_embedding)
            
            kw_score = normalized_kw_scores[i]
            
            # Hybrid Score
            hybrid_score = (self.alpha * vector_score) + ((1 - self.alpha) * kw_score)
            hybrid_scores.append((hybrid_score, chunk))
            
        # Sort by hybrid score
        hybrid_scores.sort(key=lambda x: x[0], reverse=True)
        
        return [chunk for score, chunk in hybrid_scores[:top_k]]
