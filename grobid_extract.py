import os
import requests
from bs4 import BeautifulSoup

# Define paths and the local Grobid server URL
PDF_DIR = "raw_pdfs"
OUTPUT_DIR = "extracted_text"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "corpus_clean.txt")
GROBID_URL = "http://localhost:8070/api/processFulltextDocument"

def extract_text_from_tei(xml_content):
    """Parses the Grobid XML and extracts only the abstract and body text."""
    soup = BeautifulSoup(xml_content, 'xml')
    extracted_text = []

    # 1. Grab the Abstract
    abstract = soup.find('abstract')
    if abstract:
        extracted_text.append(abstract.get_text(separator=' ', strip=True))

    # 2. Grab the Body Paragraphs (ignoring references!)
    body = soup.find('body')
    if body:
        for div in body.find_all('div'):
            # Some divs are formulas or tables, we just want the paragraph text
            for p in div.find_all('p'):
                extracted_text.append(p.get_text(separator=' ', strip=True))

    return "\n\n".join(extracted_text)

def process_with_grobid():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    all_text = []
    pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswith('.pdf')]
    
    print(f"Found {len(pdf_files)} PDFs. Sending to local Grobid server...\n")
    
    for filename in pdf_files:
        filepath = os.path.join(PDF_DIR, filename)
        print(f"Processing: {filename}...")
        
        try:
            # Send the PDF to the Grobid API
            with open(filepath, 'rb') as f:
                files = {'input': (filename, f, 'application/pdf')}
                # We ask Grobid to process the text, but skip extracting the heavy reference lists to save time
                data = {'consolidateHeader': '0', 'consolidateCitations': '0'}
                
                response = requests.post(GROBID_URL, files=files, data=data)
            
            if response.status_code == 200:
                # Parse the returned XML
                clean_text = extract_text_from_tei(response.text)
                all_text.append(clean_text)
                print(f"✅ Success!")
            else:
                print(f"❌ Server returned error code: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Failed to connect or process: {e}")

    # Save the combined, perfectly clean corpus
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write("\n\n---NEW PAPER---\n\n".join(all_text))
        
    print(f"\n🎉 Grobid extraction complete! Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    process_with_grobid()