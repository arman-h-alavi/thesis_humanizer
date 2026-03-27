import json
import os

INPUT_FILE = "extracted_text/corpus_clean.txt"
OUTPUT_FILE = "extracted_text/train_dataset.jsonl"

# Hyperparameters for chunking
CHUNK_SIZE = 1500  # roughly 300-400 words per chunk
OVERLAP = 200      # characters to overlap to preserve context context

def create_jsonl_dataset():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: Cannot find {INPUT_FILE}")
        return

    print("Loading corpus...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        full_text = f.read()

    # Clean up any remaining massive whitespace gaps
    full_text = " ".join(full_text.split())
    
    total_chars = len(full_text)
    print(f"Total characters to process: {total_chars}")

    chunks = []
    start = 0
    
    # Sliding window chunking
    while start < total_chars:
        end = min(start + CHUNK_SIZE, total_chars)
        chunk_text = full_text[start:end]
        
        # We format it exactly how the Hugging Face 'datasets' library expects
        chunks.append({"text": chunk_text})
        
        # Move the window forward, minus the overlap
        start += (CHUNK_SIZE - OVERLAP)

    print(f"Generated {len(chunks)} overlapping chunks.")

    # Save to JSONL
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')

    print(f"🎉 Dataset ready for Colab! Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    create_jsonl_dataset()