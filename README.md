# Memory Machines — Lincoln Scraping & AI Judge Evals

An end-to-end AI pipeline that scrapes historical sources about Abraham Lincoln, extracts structured claims about key events, and uses a DSPy-powered LLM Judge to evaluate the consistency between Lincoln's own primary accounts and secondary biographical sources.

---

## 🧠 What It Does

The pipeline answers the question: **"Does what biographers say about Lincoln match what Lincoln himself wrote?"**

It covers five landmark events:
- **Election Night 1860**
- **Fort Sumter Decision**
- **Gettysburg Address**
- **Second Inaugural Address**
- **Ford's Theatre Assassination**

For each event, the pipeline:
1. Scrapes primary sources (Lincoln's letters and speeches from the Library of Congress) and secondary sources (biographies from Project Gutenberg).
2. Retrieves the most relevant text chunks using a hybrid keyword + semantic embedding approach.
3. Extracts structured claims, dates, and tone from each chunk using Gemini.
4. Passes the extracted claims to a DSPy-based LLM Judge that scores consistency (0–100) and identifies contradictions.
5. Validates the Judge's reliability using Cohen's Kappa (against human labels) and self-consistency tests.

---

## 📂 Repository Structure

```
.
├── main.py                     # Step 1: Scrape & normalize data
├── extract_events_hybrid.py    # Step 2: Retrieve & extract claims
├── evaluate_consistency.py     # Step 3: Run the LLM Judge
├── validate_judge.py           # Step 4: Validate Judge reliability
├── compare_prompts.py          # Step 5: Compare prompting strategies
├── dashboard.py                # Streamlit dashboard for visualization
│
├── src/
│   ├── scrapers/
│   │   ├── gutenberg.py        # Scraper for Project Gutenberg biographies
│   │   └── loc.py              # Scraper for Library of Congress letters/speeches
│   ├── pipeline/
│   │   ├── hybrid_retriever.py # Hybrid retrieval (keywords + text-embedding-004)
│   │   ├── extractor.py        # Gemini-based claim extractor
│   │   ├── judge.py            # DSPy LLM Judge + statistical validators
│   │   └── retriever.py        # Keyword-only retriever (legacy)
│   ├── models/
│   │   └── schema.py           # Pydantic document schema
│   └── utils/
│
├── data/
│   ├── normalized/             # Output of main.py (scraped & cleaned JSON)
│   ├── extracted/              # Output of extract_events_hybrid.py
│   │   └── events_hybrid.json  # Extracted claims per event per source
│   ├── evaluated/              # Output of evaluate/validate scripts
│   │   ├── consistency_report.json
│   │   ├── kappa_report.json
│   │   ├── full_consistency_report.json
│   │   └── prompt_comparison_report.json
│   └── manual_labels.json      # Human-labeled pairs for Kappa validation
│
└── requirements.txt
```

---

## 🚀 Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/ghantasala-sr/Lincoln-Scraping-AI-Judge-Evals.git
cd Lincoln-Scraping-AI-Judge-Evals
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file in the root directory:

```env
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-2.5-flash   # Optional — defaults to gemini-2.5-flash
```

> Get a free Gemini API key at [aistudio.google.com](https://aistudio.google.com/).

---

## ▶️ Running the Full Pipeline

Run the steps in order:

### Step 1 — Scrape & normalize sources

```bash
python main.py
```

Scrapes 5 Project Gutenberg biographies and 5 Library of Congress documents (letters, speeches, the Gettysburg Address). Saves normalized JSON to `data/normalized/`.

### Step 2 — Extract claims using hybrid retrieval

```bash
python extract_events_hybrid.py
```

For each document and each of the 5 events, uses a hybrid retriever (keyword match + Gemini `text-embedding-004` semantic embeddings) to find the most relevant chunks, then calls Gemini to extract structured claims, dates, and tone. Output: `data/extracted/events_hybrid.json`.

### Step 3 — Evaluate consistency with the LLM Judge

```bash
python evaluate_consistency.py
```

Groups extracted claims by event, pairs Lincoln's primary sources against secondary sources, and runs each pair through the DSPy-based `EventJudge`. The Judge uses Chain-of-Thought reasoning to assign a consistency score (0–100) and classify contradictions as `factual`, `interpretive`, or `omission` with severity levels (`minor`, `moderate`, `major`). Output: `data/evaluated/consistency_report.json`.

### Step 4 — Validate Judge reliability

```bash
python validate_judge.py
```

Runs two statistical validation experiments:
- **Cohen's Kappa** — compares LLM judgments against human-labeled pairs in `data/manual_labels.json` to measure inter-rater agreement.
- **Self-Consistency** — runs the same comparison 5 times at `temperature=0.7` and measures score variance (std dev) to test stability.

Outputs: `data/evaluated/kappa_report.json`, `data/evaluated/full_consistency_report.json`.

### Step 5 — Compare prompting strategies

```bash
python compare_prompts.py
```

Benchmarks three prompt strategies — **Zero-Shot**, **Chain-of-Thought (CoT)**, and **Few-Shot** — by measuring score variance (lower std dev = more stable). Outputs a comparison report to `data/evaluated/prompt_comparison_report.json`.

---

## 📊 Interactive Dashboard

The easiest way to explore the results is the Streamlit dashboard:

```bash
streamlit run dashboard.py
```

The dashboard lets you:
- **Inspect raw & normalized data** from Gutenberg and LoC.
- **View extracted claims** per event and source.
- **Explore consistency scores and contradictions** found by the Judge.
- **Examine validation results** — Kappa scores, self-consistency charts, and prompt comparisons.

---

## 🔬 Key Components

### LLM Judge (`src/pipeline/judge.py`)

Built with [DSPy](https://github.com/stanfordnlp/dspy) using a `ChainOfThought` module over a `CompareAccounts` signature. The Judge is instructed to:
- Use **only** text provided (no outside knowledge).
- Distinguish **factual** contradictions (dates, numbers, locations) from **interpretive** differences (tone, motivation) and **omissions**.
- Return structured JSON with `consistency_score`, `contradictions[]`, `reasoning`, and `confidence`.

### Hybrid Retriever (`src/pipeline/hybrid_retriever.py`)

Combines two signals with a configurable blend (`alpha=0.7` by default):
- **Keyword score** — BM25-style match against event-specific keyword lists.
- **Semantic score** — cosine similarity using Google's `text-embedding-004` model.

### Consistency Score Interpretation

| Score | Meaning |
|-------|---------|
| 90–100 | Near-identical accounts, only wording differs |
| 70–89 | Same narrative, minor factual discrepancies |
| 50–69 | Moderate differences, some omissions |
| 30–49 | Significant contradictions |
| 0–29 | Fundamentally conflicting accounts |

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `google-generativeai` | Gemini LLM for extraction and embedding |
| `dspy-ai` | DSPy framework for the LLM Judge |
| `scikit-learn` | Cohen's Kappa and confusion matrix |
| `numpy` | Score statistics |
| `streamlit` + `plotly` + `pandas` | Dashboard visualization |
| `beautifulsoup4` + `lxml` | Web scraping |
| `pydantic` | Data validation and schemas |
| `python-dotenv` | Environment variable management |
