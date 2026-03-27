import os
import fitz  # PyMuPDF
import re

# Define paths
PDF_DIR = "raw_pdfs"
OUTPUT_DIR = "extracted_text"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "corpus_raw.txt")

def clean_text(text):
    """Basic cleaning to remove garbage text from PDFs."""
    # Remove excessive newlines and spaces
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r' +', ' ', text)
    
    # Filter out lines that are just numbers (usually page numbers) or very short
    cleaned_lines = []
    for line in text.split('\n'):
        line = line.strip()
        # Keep the line if it has more than 3 words (filters out random headers/footers)
        if len(line.split()) > 3:
            cleaned_lines.append(line)
            
    return " ".join(cleaned_lines)

def process_pdfs():
    # Ensure output directory exists
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    all_text = []
    pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswith('.pdf')]
    
    print(f"Found {len(pdf_files)} PDFs. Starting extraction...\n")
    
    for filename in pdf_files:
        filepath = os.path.join(PDF_DIR, filename)
        try:
            # Open the PDF
            doc = fitz.open(filepath)
            doc_text = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                # Extract text preserving blocks helps with reading flow
                text = page.get_text("text") 
                cleaned = clean_text(text)
                doc_text.append(cleaned)
                
            full_doc_text = "\n\n".join(doc_text)
            all_text.append(full_doc_text)
            
            print(f"✅ Successfully extracted: {filename} ({len(doc)} pages)")
            
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")

    # Save to a single corpus file
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write("\n\n---NEW PAPER---\n\n".join(all_text))
        
    print(f"\n🎉 Extraction complete! Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    process_pdfs()