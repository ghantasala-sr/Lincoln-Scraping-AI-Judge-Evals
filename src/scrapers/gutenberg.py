import requests
from bs4 import BeautifulSoup
import re
from typing import Optional
from src.models.schema import Document
from src.utils.text_processing import clean_text, clean_html_artifacts

class GutenbergScraper:
    def __init__(self):
        self.base_url = "https://www.gutenberg.org"

    def scrape(self, url: str) -> Document:
        print(f"Scraping Gutenberg URL: {url}")
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract Metadata
        metadata = self._extract_metadata(soup)
        
        # Construct ID from URL
        doc_id = f"gutenberg_{url.rstrip('/').split('/')[-1]}"

        # Extract Content
        content_url = self._find_txt_link(soup, url)
        content = self._fetch_content(content_url, doc_id)

        return Document(
            id=doc_id,
            title=metadata.get('Title', 'Unknown Title'),
            reference=url,
            document_type="Book",
            date=metadata.get('Release Date', 'Unknown Date'),
            from_=metadata.get('Author', 'Unknown Author'),
            content=content
        )

    def _extract_metadata(self, soup: BeautifulSoup) -> dict:
        metadata = {}
        bibrec = soup.find('div', id='bibrec')
        if not bibrec:
            return metadata
        
        table = bibrec.find('table', class_='bibrec')
        if not table:
            return metadata

        for row in table.find_all('tr'):
            th = row.find('th')
            td = row.find('td')
            if th and td:
                key = th.get_text(strip=True).replace(':', '')
                value = td.get_text(strip=True)
                metadata[key] = value
        
        # Clean up date
        if 'Release Date' in metadata:
            # format often "Month, Year [EBook #...]"
            metadata['Release Date'] = metadata['Release Date'].split('[')[0].strip()

        return metadata

    def _find_txt_link(self, soup: BeautifulSoup, base_url: str) -> str:
        # Look for the link to Plain Text UTF-8
        link = soup.find('a', string=re.compile(r'Plain Text UTF-8'))
        if link:
            href = link.get('href')
            if href.startswith('//'):
                return 'https:' + href
            elif href.startswith('/'):
                return self.base_url + href
            return href
        
        # Fallback: try to construct it or find other text links
        # Usually /files/ID/ID-0.txt or similar
        raise ValueError("Could not find Plain Text UTF-8 link")

    def _fetch_content(self, url: str, doc_id: str) -> str:
        print(f"Fetching content from: {url}")
        response = requests.get(url)
        response.raise_for_status()
        response.encoding = 'utf-8' 
        
        # Save raw content
        import os
        raw_path = os.path.join("data/raw", f"{doc_id}.txt")
        with open(raw_path, 'w', encoding='utf-8') as f:
            f.write(response.text)
            
        return clean_text(response.text)
