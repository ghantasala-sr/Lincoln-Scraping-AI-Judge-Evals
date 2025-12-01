import re

def clean_text(text: str) -> str:
    """
    Basic text cleaning while preserving structure.
    Removes excessive whitespace but keeps paragraph breaks.
    """
    if not text:
        return ""
    # Normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    # Remove trailing whitespace on lines
    lines = [line.rstrip() for line in text.split('\n')]
    # Join back
    return '\n'.join(lines)

def clean_html_artifacts(text: str) -> str:
    """
    Removes specific HTML-like keywords and navigation text.
    """
    if not text:
        return ""
    
    # Common navigation/UI terms in LoC/Gutenberg
    artifacts = [
        r"skip navigation",
        r"Library of Congress",
        r"Ask a Librarian",
        r"Digital Collections",
        r"Library Catalogs",
        r"\bSearch\b",
        r"\bGO\b",
        r"Back to Exhibition",
        r"Transcription",
        r"Connect with the Library",
        r"All ways to connect",
        r"Find Us On",
        r"Subscribe & Comment",
        r"RSS & E-Mail",
        r"\bBlogs\b",
        r"Download & Play",
        r"\bApps\b",
        r"\bPodcasts\b",
        r"\bWebcasts\b",
        r"iTunesU",
        r"\bQuestions\b",
        r"Contact Us",
        r"About \|",
        r"Press \|",
        r"Jobs \|",
        r"\bDonate\b",
        r"Inspector General \|",
        r"Legal \|",
        r"Accessibility \|",
        r"External Link Disclaimer \|",
        r"USA\.gov",
        r"Speech Enabled",
        r"Transcribed and reviewed by contributors participating in the By The People project at crowd\.loc\.gov\."
    ]
    
    cleaned = text
    for artifact in artifacts:
        cleaned = re.sub(artifact, "", cleaned, flags=re.IGNORECASE)
        
    # Remove multiple newlines created by removal
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned.strip()

def extract_metadata_from_title(title: str) -> dict:
    """
    Extracts 'to', 'from', and 'place' from title using regex heuristics.
    Returns a dict with keys 'to', 'from', 'place'.
    """
    metadata = {'to': None, 'from': None, 'place': None}
    
    # Pattern: "Collection Info: Person A to Person B, Date"
    # User request: "scrap whatever is from : .... to"
    
    # Regex explanation:
    # (?:.*:\s*)?  -> Non-capturing group for prefix ending in colon. .* is greedy, so it takes up to the LAST colon.
    # (.*?)        -> Group 1: Sender (non-greedy, stops at " to ")
    # \s+to\s+     -> Delimiter " to "
    # (.*?),       -> Group 2: Recipient (non-greedy, stops at comma)
    
    to_match = re.search(r"(?:.*:\s*)?(.*?)\s+to\s+(.*?),", title)
    if to_match:
        metadata['from'] = to_match.group(1).strip()
        metadata['to'] = to_match.group(2).strip()
        
    # Try to extract place if present (often not in title, but sometimes "Place, Date")
    # This is harder from title alone without a clear delimiter, usually place is in the content or separate field.
    # We will leave place as None for now unless a clear pattern emerges.
    
    return metadata


def classify_document_type(title: str, content: str) -> str:
    """
    Classifies document type based on keywords in title and content.
    """
    title_lower = title.lower()
    content_lower = content.lower()[:500] # Check first 500 chars
    
    if "speech" in title_lower or "address" in title_lower:
        return "Speech"
    if "proclamation" in title_lower:
        return "Proclamation"
    if "letter" in title_lower or " to " in title_lower: # "X to Y" implies letter
        return "Letter"
    if "telegram" in title_lower:
        return "Telegram"
    if "memorandum" in title_lower or "memo" in title_lower:
        return "Memorandum"
    if "note" in title_lower:
        return "Unknown"
        
    return "Manuscript" # Default

# --- Place Extraction Logic (User Provided) ---

# US State abbreviations mapping
STATE_ABBREVS = {
    'Ala': 'Alabama', 'Alab': 'Alabama', 'AL': 'Alabama',
    'Ariz': 'Arizona', 'AZ': 'Arizona',
    'Ark': 'Arkansas', 'AR': 'Arkansas',
    'Cal': 'California', 'Calif': 'California', 'CA': 'California',
    'Colo': 'Colorado', 'CO': 'Colorado',
    'Conn': 'Connecticut', 'CT': 'Connecticut',
    'Del': 'Delaware', 'DE': 'Delaware',
    'Fla': 'Florida', 'FL': 'Florida',
    'Ga': 'Georgia', 'GA': 'Georgia',
    'Ill': 'Illinois', 'IL': 'Illinois',
    'Ind': 'Indiana', 'IN': 'Indiana',
    'Ia': 'Iowa', 'IA': 'Iowa',
    'Kan': 'Kansas', 'Kans': 'Kansas', 'KS': 'Kansas',
    'Ky': 'Kentucky', 'KY': 'Kentucky',
    'La': 'Louisiana', 'LA': 'Louisiana',
    'Me': 'Maine', 'ME': 'Maine',
    'Md': 'Maryland', 'MD': 'Maryland',
    'Mass': 'Massachusetts', 'MA': 'Massachusetts',
    'Mich': 'Michigan', 'MI': 'Michigan',
    'Minn': 'Minnesota', 'MN': 'Minnesota',
    'Miss': 'Mississippi', 'MS': 'Mississippi',
    'Mo': 'Missouri', 'MO': 'Missouri',
    'Mont': 'Montana', 'MT': 'Montana',
    'Neb': 'Nebraska', 'Nebr': 'Nebraska', 'NE': 'Nebraska',
    'Nev': 'Nevada', 'NV': 'Nevada',
    'NH': 'New Hampshire',
    'NJ': 'New Jersey',
    'NM': 'New Mexico',
    'NY': 'New York',
    'NC': 'North Carolina',
    'ND': 'North Dakota',
    'O': 'Ohio', 'OH': 'Ohio',
    'Okla': 'Oklahoma', 'OK': 'Oklahoma',
    'Ore': 'Oregon', 'Oreg': 'Oregon', 'OR': 'Oregon',
    'Pa': 'Pennsylvania', 'Penn': 'Pennsylvania', 'Penna': 'Pennsylvania', 'PA': 'Pennsylvania',
    'RI': 'Rhode Island',
    'SC': 'South Carolina', 'S C': 'South Carolina', 'S. C': 'South Carolina',
    'SD': 'South Dakota',
    'Tenn': 'Tennessee', 'TN': 'Tennessee',
    'Tex': 'Texas', 'TX': 'Texas',
    'Vt': 'Vermont', 'VT': 'Vermont',
    'Va': 'Virginia', 'VA': 'Virginia',
    'Wash': 'Washington', 'WA': 'Washington',
    'WV': 'West Virginia', 'W Va': 'West Virginia', 'W. Va': 'West Virginia',
    'Wis': 'Wisconsin', 'Wisc': 'Wisconsin', 'WI': 'Wisconsin',
    'Wyo': 'Wyoming', 'WY': 'Wyoming',
    'DC': 'D.C.', 'D C': 'D.C.', 'D. C': 'D.C.'
}

# Known US cities for context
KNOWN_CITIES = [
    'Washington', 'Springfield', 'Charleston', 'Richmond', 'Philadelphia',
    'New York', 'Boston', 'Baltimore', 'Gettysburg', 'Atlanta', 'Chicago',
    'St Louis', 'Cincinnati', 'Cleveland', 'Detroit', 'Pittsburgh',
    'New Orleans', 'Wheeling', 'Nashville', 'Memphis', 'Louisville'
]

# Month patterns for dateline detection
MONTHS = r'(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)'


def normalize_state(abbrev):
    """Convert state abbreviation to full name."""
    # Clean up the abbreviation
    cleaned = abbrev.strip().rstrip('.')
    
    # Try direct match
    if cleaned in STATE_ABBREVS:
        return STATE_ABBREVS[cleaned]
    
    # Try without periods: "S. C." -> "S C"
    no_periods = cleaned.replace('.', '').strip()
    if no_periods in STATE_ABBREVS:
        return STATE_ABBREVS[no_periods]
    
    # Try uppercase
    if cleaned.upper() in STATE_ABBREVS:
        return STATE_ABBREVS[cleaned.upper()]
    
    return None


def infer_state_from_context(text, city):
    """Try to infer state from surrounding context."""
    # Look for state mentions near the city
    context_patterns = [
        (r'West\s+Virginia', 'West Virginia'),
        (r'Virginia', 'Virginia'),
        (r'Pennsylvania', 'Pennsylvania'),
        (r'Illinois', 'Illinois'),
        (r'South\s+Carolina', 'South Carolina'),
        (r'North\s+Carolina', 'North Carolina'),
    ]
    
    for pattern, state in context_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return state
    
    # City-to-state mapping for common cities
    city_states = {
        'Wheeling': 'West Virginia',
        'Charleston': 'South Carolina',  # In Civil War context
        'Springfield': 'Illinois',  # Lincoln's home
        'Gettysburg': 'Pennsylvania',
        'Richmond': 'Virginia',
        'Washington': 'D.C.'
    }
    
    return city_states.get(city)


def extract_place_from_content(content: str) -> str:
    """
    Extract place from document content using dateline patterns.
    19th century letters typically start with: City, State Date
    """
    if not content:
        return None
        
    # Only search first 1000 chars (datelines are at the start)
    header = content[:1000]
    
    # Pattern 1: "City State Month Day Year" (e.g., "Springfield Ill Nov 10th 1860")
    pattern1 = rf'([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)\s+([A-Z][a-z]{{1,4}}\.?)\s+{MONTHS}\.?\s+\d{{1,2}}'
    match = re.search(pattern1, header)
    if match:
        city = match.group(1)
        state_abbrev = match.group(2)
        state = normalize_state(state_abbrev)
        if state and city in KNOWN_CITIES:
            return f"{city}, {state}"
    
    # Pattern 2: "City S. C. Month" (e.g., "Charleston S. C. April")
    pattern2 = rf'([A-Z][a-z]+)\s+([A-Z]\.?\s*[A-Z]\.?)\s+{MONTHS}'
    match = re.search(pattern2, header)
    if match:
        city = match.group(1)
        state_abbrev = match.group(2)
        state = normalize_state(state_abbrev)
        if state:
            return f"{city}, {state}"
    
    # Pattern 3: "City, Month Day, Year" (e.g., "Wheeling, March 18, 1865")
    pattern3 = rf'([A-Z][a-z]+),\s+{MONTHS}\s+\d{{1,2}},?\s+\d{{4}}'
    match = re.search(pattern3, header)
    if match:
        city = match.group(1)
        if city in KNOWN_CITIES:
            # Try to find state context nearby
            state = infer_state_from_context(header, city)
            if state:
                return f"{city}, {state}"
            return city
    
    # Pattern 4: "City, State\n" or "City, State," in letterhead
    pattern4 = r'([A-Z][a-z]+),?\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)[,\n]'
    match = re.search(pattern4, header)
    if match:
        city = match.group(1)
        potential_state = match.group(2)
        # Check if second word is actually a state name
        if potential_state in STATE_ABBREVS.values():
            return f"{city}, {potential_state}"
    
    return None

def extract_place_from_title_heuristics(title: str) -> str:
    """Extract place from document title as fallback."""
    if not title:
        return None
        
    title_lower = title.lower()
    
    # Look for location keywords in title
    if 'gettysburg' in title_lower:
        return "Gettysburg, Pennsylvania"
    
    if 'inaugural address' in title_lower or 'second inaugural' in title_lower:
        return "Washington, D.C."
    
    if 'first inaugural' in title_lower:
        return "Washington, D.C."
    
    return None
