import requests
from bs4 import BeautifulSoup
import time
import re
from src.models.schema import Document
from src.utils.text_processing import clean_text, clean_html_artifacts, extract_metadata_from_title, classify_document_type, extract_place_from_content, extract_place_from_title_heuristics

class LocScraper:
    def __init__(self):
        self.base_url = "https://www.loc.gov"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'
        }

    def scrape(self, url: str) -> Document:
        print(f"Scraping LoC URL: {url}")
        time.sleep(1)  # Respect rate limits

        # Handle different URL types
        if "exhibits/gettysburg-address" in url:
            return self._scrape_exhibit(url)
        
        # Default to API/HTML hybrid for items and resources
        return self._scrape_item(url)

    def _scrape_item(self, url: str) -> Document:
        # Use JSON API for metadata
        api_url = f"{url}?fo=json"
        try:
            response = requests.get(api_url, headers=self.headers)
            response.raise_for_status()
            data = response.json()
            
            # Extract ID
            doc_id = f"loc_{url.rstrip('/').split('/')[-1]}"

            # Save raw JSON
            import json
            import os
            raw_path = os.path.join("data/raw", f"{doc_id}.json")
            with open(raw_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            
            # Extract metadata
            item = data.get('item', {})
            if not item:
                item = data # Sometimes root is the item
            
            
            title = item.get('title', 'Unknown Title')
            if isinstance(title, list):
                title = title[0]
                
            date = item.get('date', 'Unknown Date')
            
            # Try to extract content from JSON resources first
            content = ""
            # Prefer root-level resources which often contain the detailed file list
            resources = data.get('resources', [])
            if not resources:
                resources = item.get('resources', [])
                
            for resource in resources:
                if 'files' in resource and isinstance(resource['files'], list):
                    for file_group in resource['files']:
                        if isinstance(file_group, list):
                            for file_item in file_group:
                                if file_item.get('use') == 'text' and 'fulltext' in file_item:
                                    content = file_item['fulltext']
                                    break
                        if content: break
                if content: break
            
            # If no content in JSON, try fulltext_file URL from resources
            if not content:
                for resource in resources:
                    if 'fulltext_file' in resource:
                        file_url = resource['fulltext_file']
                        print(f"Fetching fulltext from: {file_url}")
                        try:
                            ft_response = requests.get(file_url, headers=self.headers)
                            ft_response.raise_for_status()
                            
                            if file_url.endswith('.txt'):
                                content = clean_text(ft_response.text)
                            else:
                                # Parse XML/HTML
                                soup_ft = BeautifulSoup(ft_response.content, 'xml') # Try XML parser first
                                content = clean_text(soup_ft.get_text())
                            
                            if content: break
                        except Exception as e:
                            print(f"Failed to fetch fulltext file {file_url}: {e}")

            # Fallback to HTML scraping if JSON content is empty (though HTML is likely blocked)
            if not content:
                print(f"Warning: No content found in JSON for {url}, trying HTML fallback...")
                html_response = requests.get(url, headers=self.headers)
                if html_response.status_code == 200:
                    soup = BeautifulSoup(html_response.content, 'html.parser')
                    content = self._extract_transcription(soup)
                else:
                    print(f"HTML fallback failed with status {html_response.status_code}")

            # Clean XML/HTML tags from content if it came from JSON
            if content and '<' in content:
                soup_content = BeautifulSoup(content, 'html.parser')
                content = clean_text(soup_content.get_text())
            
            # Advanced cleaning
            content = clean_html_artifacts(content)
            
            # Extract metadata from title
            title_meta = extract_metadata_from_title(title)
            
            # Extract place: try content first, then title heuristics, then title metadata
            place = extract_place_from_content(content)
            if not place:
                place = extract_place_from_title_heuristics(title)
            if not place:
                place = title_meta['place']
            
            # Classify document type
            doc_type = classify_document_type(title, content)
            
            # Create document object
            doc = Document(
                id=doc_id,
                title=title,
                reference=url,
                document_type=doc_type,
                date=str(date),
                place=place,
                from_=title_meta['from'],
                to=title_meta['to'],
                content=content
            )
            
            return doc
            
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            # Return a partial document or raise
            return Document(
                id=f"loc_error_{int(time.time())}",
                title="Error Scraping",
                reference=url,
                document_type="Error",
                date="",
                content=""
            )

    def _scrape_exhibit(self, url: str) -> Document:
        response = requests.get(url, headers=self.headers)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Specific extraction for the exhibit page
        content = clean_text(soup.get_text()) # Fallback
        
        # Try to find specific transcription area
        # The URL provided is .../trans-nicolay-copy.html, which is likely the transcription itself
        # So we can probably just take the body text
        body = soup.find('body')
        if body:
            content = clean_text(body.get_text())
            
        # Advanced cleaning
        content = clean_html_artifacts(content)
        
        title = "Gettysburg Address (Nicolay Copy)"
        doc_type = classify_document_type(title, content)

        return Document(
            id="loc_gettysburg_nicolay",
            title=title,
            reference=url,
            document_type=doc_type,
            date="1863",
            content=content
        )

    def _extract_transcription(self, soup: BeautifulSoup) -> str:
        # Heuristics for finding transcription on LoC pages
        # 1. Look for 'Transcription' tab or section
        # 2. Look for <div id="transcription"> or similar
        
        # This is a simplified heuristic; actual LoC pages vary
        transcription_div = soup.find('div', {'id': 'transcription-text'})
        if transcription_div:
            return clean_text(transcription_div.get_text())
            
        # Try finding text within a specific container
        # Often LoC puts transcription in a <section> or <div> with specific classes
        # For now, let's try to grab the main text if specific ID fails
        # or return empty string to indicate manual check needed
        return ""
