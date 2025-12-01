import re
from typing import List, Dict

class ContextRetriever:
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap
        
        # Keywords for the 5 key events
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

    def retrieve(self, text: str, event_name: str, top_k: int = 3) -> List[str]:
        """
        Retrieves the most relevant chunks for a given event.
        Scoring is based on keyword density.
        """
        if event_name not in self.event_keywords:
            return []
            
        keywords = self.event_keywords[event_name]
        chunks = self.chunk_text(text)
        
        scored_chunks = []
        for chunk in chunks:
            score = 0
            chunk_lower = chunk.lower()
            for keyword in keywords:
                if keyword.lower() in chunk_lower:
                    score += 1
            
            if score > 0:
                scored_chunks.append((score, chunk))
        
        # Sort by score descending
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        
        # Return top K chunks
        return [chunk for score, chunk in scored_chunks[:top_k]]
