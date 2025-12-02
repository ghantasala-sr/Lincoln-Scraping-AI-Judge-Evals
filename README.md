# Memory_Machines_Lincoln_Project

A robust AI pipeline to extract, evaluate, and validate historical events from the life of Abraham Lincoln, comparing his own primary accounts against secondary biographical sources.

## 📂 Project Structure

### **Core Pipeline**
*   `src/scrapers/`: Scrapers for Project Gutenberg (Biographies) and Library of Congress (Letters).
*   `src/pipeline/hybrid_retriever.py`: Hybrid retrieval system (Keywords + Embeddings) to find relevant text chunks.
*   `src/pipeline/extractor.py`: LLM-based extraction of claims, dates, and tone.
*   `src/pipeline/judge.py`: **The LLM Judge** (DSPy-based) that evaluates consistency between sources.

### **Orchestration Scripts**
*   `main.py`: Runs the scrapers and normalizes data into `data/normalized/`.
*   `extract_events_hybrid.py`: Runs the hybrid retrieval and extraction pipeline.
*   `evaluate_consistency.py`: Runs the LLM Judge to generate consistency reports.
*   `validate_judge.py`: Runs statistical validation (Kappa, Self-Consistency).
*   `compare_prompts.py`: Compares Zero-Shot vs. CoT vs. Few-Shot strategies.

### **Visualization**
*   `dashboard.py`: **Streamlit Dashboard** to visualize the entire pipeline.

### **Data**
*   `data/normalized/`: Cleaned JSON data from scrapers.
*   `data/extracted/`: Extracted events (`events_hybrid.json`).
*   `data/evaluated/`: Evaluation reports (`consistency_report.json`, `kappa_report.json`, etc.).

---

## 🚀 Setup & Installation

1.  **Clone the repository**
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Environment Configuration**:
    Create a `.env` file in the root directory with your Gemini API key:
    ```env
    GEMINI_API_KEY=your_api_key_here
    GEMINI_MODEL=gemini-2.5-flash  # Optional, defaults to this
    ```

---

## 📊 How to See the Data

The best way to explore the data and results is through the **Interactive Dashboard**.

### **Run the Dashboard**
```bash
streamlit run dashboard.py
```

This will open a web interface where you can:
1.  **Inspect Data**: View raw and normalized text from Gutenberg and LoC.
2.  **View Extraction**: See the claims and metadata extracted for events like "Fort Sumter".
3.  **Analyze Evaluation**: Explore the consistency scores and contradictions found by the Judge.
4.  **Check Validation**: See the statistical proof of the Judge's reliability (Kappa & Stability tests).

### **Manual Inspection**
You can also directly inspect the JSON files in the `data/` directory:
*   `data/extracted/events_hybrid.json`: The raw extracted event data.
*   `data/evaluated/consistency_report.json`: The detailed judge outputs.
