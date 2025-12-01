import json
import os
from typing import List
from src.scrapers.gutenberg import GutenbergScraper
from src.scrapers.loc import LocScraper
from src.models.schema import Document

# Constants
GUTENBERG_URLS = [
    "https://www.gutenberg.org/ebooks/6812",
    "https://www.gutenberg.org/ebooks/6811",
    "https://www.gutenberg.org/ebooks/12801/",
    "https://www.gutenberg.org/ebooks/14004/",
    "https://www.gutenberg.org/ebooks/18379"
]

LOC_URLS = [
    "https://www.loc.gov/item/mal0440500/",
    "https://www.loc.gov/resource/mal.0882800",
    "https://www.loc.gov/exhibits/gettysburg-address/ext/trans-nicolay-copy.html",
    "https://www.loc.gov/resource/mal.4361300",
    "https://www.loc.gov/resource/mal.4361800/"
]

DATA_DIR = "data/normalized"

def save_dataset(documents: List[Document], filename: str):
    filepath = os.path.join(DATA_DIR, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        # Dump list of dicts
        data = [doc.model_dump(by_alias=True) for doc in documents]
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(documents)} items to {filepath}")

def main():
    # Ensure directories exist
    os.makedirs(DATA_DIR, exist_ok=True)

    # 1. Process Gutenberg
    print("--- Processing Project Gutenberg ---")
    gutenberg_scraper = GutenbergScraper()
    gutenberg_docs = []
    for url in GUTENBERG_URLS:
        try:
            doc = gutenberg_scraper.scrape(url)
            gutenberg_docs.append(doc)
            print(f"Successfully scraped: {doc.title}")
        except Exception as e:
            print(f"Failed to scrape {url}: {e}")
    
    save_dataset(gutenberg_docs, "gutenberg_dataset.json")

    # 2. Process Library of Congress
    print("\n--- Processing Library of Congress ---")
    loc_scraper = LocScraper()
    loc_docs = []
    for url in LOC_URLS:
        try:
            doc = loc_scraper.scrape(url)
            loc_docs.append(doc)
            print(f"Successfully scraped: {doc.title}")
        except Exception as e:
            print(f"Failed to scrape {url}: {e}")
            import traceback
            traceback.print_exc()

    save_dataset(loc_docs, "loc_dataset.json")

if __name__ == "__main__":
    main()
