# Academic Style-Transfer Pipeline (Llama-3 QLoRA)

## Overview
This project is an end-to-end NLP pipeline designed to perform stylistic text transformation. It fine-tunes a foundational Large Language Model (Meta-Llama-3-8B) to translate standard drafted text into highly formal, domain-specific academic prose. The model's target statistical distribution is tuned on peer-reviewed literature focusing on applied statistics, historical price data analysis, and metadata formatting.

By utilizing a local, fine-tuned LoRA (Low-Rank Adaptation) adapter, this tool functions entirely offline during inference, ensuring complete data privacy for unpublished thesis drafts and academic research.

*Note: This project deliberately utilizes QLoRA fine-tuning rather than RAG. While RAG is optimal for factual retrieval, it cannot fundamentally alter the statistical burstiness or perplexity of the generator's prose. Fine-tuning was required to securely capture the domain-specific stylometry.*

## Architecture & Workflow
The project is divided into three distinct phases:

### Phase 1: Data Extraction (Docker & Grobid)
* **Objective:** Extract pure narrative prose from complex academic PDFs while filtering out headers, footers, and reference lists.
* **Tooling:** A local `Grobid` machine learning server containerized via Docker.
* **Process:** PDFs are sent via a Python API to the Grobid server, which parses the documents layout-by-layout and returns structured XML (TEI) containing only the abstract and body paragraphs.

### Phase 2: Dataset Preparation (Sliding Window Chunking)
* **Objective:** Format the raw text into a deep-learning-compatible format without destroying the micro-flow of the academic transitions.
* **Process:** The raw text is tokenized and divided into overlapping chunks (1500 characters with a 200-character overlap) to preserve contextual chain-links. The output is formatted as a `.jsonl` file perfectly optimized for Hugging Face's `datasets` library.

### Phase 3: Deep Learning & Fine-Tuning (QLoRA)
* **Objective:** Shift the base Llama-3 model's weights toward the target academic style using a highly constrained hardware environment (NVIDIA T4 GPU, 16GB VRAM).
* **Technique:** * The base model is compressed to 4-bit precision (`nf4`) using `BitsAndBytes`.
  * A LoRA adapter (Rank=16) is attached to the attention modules (`q_proj`, `k_proj`, `v_proj`, `o_proj`).
  * The model is trained via Causal Language Modeling (Next-Token Prediction) using the `trl` library's `SFTTrainer`.

## Hardware Constraints & Troubleshooting

**NVIDIA T4 GPU vs. BFloat16 Architecture**
Llama-3 natively utilizes the `bfloat16` data type. However, free-tier cloud environments often rely on older NVIDIA Turing architectures (like the T4 GPU), which physically lack the silicon pathways to compute `bfloat16` gradients natively. 

Attempting to train Llama-3 with standard Hugging Face configurations on a T4 results in a PyTorch AMP GradScaler crash:
`NotImplementedError: "_amp_foreach_non_finite_check_and_unscale_cuda" not implemented for 'BFloat16'`

**The Implemented Fix:**
To bypass this hardware mismatch without relying on incredibly slow software emulation, this pipeline's training script forces a total environment override:
1. `fp16=False` and `bf16=False` in the `SFTConfig` to completely disable the buggy Automatic Mixed Precision (AMP) GradScaler.
2. The model loader forces `torch.float16`.
3. The LoRA adapter is trained in standard 32-bit math, which fits comfortably within the T4's VRAM memory ceiling while maintaining fast iteration speeds.

### Phase 4: Instruct Architecture & Kaggle Migration
* **The Pivot:** Migrated from the `Meta-Llama-3-8B` base model to the `Instruct` variant to eliminate prompt hallucinations and enforce `<|eot_id|>` stop tokens via native Chat Templates.
* **Hardware Upgrade:** Moved the pipeline to Kaggle using Dual T4 GPUs (32GB VRAM). This overhead allowed us to reverse the lossy 4-bit compression and run **8-bit inference**, preserving the complex reasoning pathways required for applied statistics.
* **The Alignment Tax (Limitation):** Despite aggressive system prompting, the Instruct model's native RLHF alignment (which favors verbose "AI-speak") actively fought the custom LoRA stylometry. It also exhibited a strong bias toward summarizing multi-paragraph inputs.

### Phase 5: Synthetic Paired Data (Current Pipeline)
To completely overwrite the base model's RLHF bias, the pipeline shifted from unpaired text ingestion to supervised translation mapping.
* **Data Generation:** A Python script was used to convert the source papers into a synthetic `[Robotic AI Draft] -> [Human Academic Target]` dataset.
* **Objective:** Retrain the LoRA adapter to explicitly learn the mathematical function of stripping AI filler and outputting dry, human-written statistics and data science prose.
## Usage
*(Add your instructions here for running the extraction, preparation, and inference scripts once we finalize them).*
