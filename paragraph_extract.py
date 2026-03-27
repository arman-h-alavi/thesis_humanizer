import os
import json

# Change this to wherever your text files are saved
INPUT_FILE = ".\extracted_text\corpus_clean.txt" 
OUTPUT_FILE = ".\extracted_text\clean_human_paragraphs.json"

MINIMUM_WORD_COUNT = 40
extracted_paragraphs = []

print(f"Scanning file: {INPUT_FILE}...")

with open(INPUT_FILE, "r", encoding="utf-8") as file:
    content = file.read()
    # Split by double line breaks to isolate paragraphs
    paragraphs = content.split("\n\n") 
        
          
    for p in paragraphs:
        # Clean up stray newlines inside the paragraph
        clean_text = p.strip().replace("\n", " ")
        word_count = len(clean_text.split())
        
        # THE FILTER: Keep only the dense, meaty paragraphs
        if word_count >= MINIMUM_WORD_COUNT:
            extracted_paragraphs.append(clean_text)

print(f"Extraction complete!")
print(f"Total high-quality paragraphs saved: {len(extracted_paragraphs)}")

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(extracted_paragraphs, f, indent=4)