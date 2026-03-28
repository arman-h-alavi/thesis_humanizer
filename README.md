# Academic Humanizer Pipeline (Llama-3 QLoRA)

[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-ee4c2c.svg)](https://pytorch.org/)
[![Library: Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers%20%7C%20PEFT%20%7C%20TRL-F9D371.svg)](https://huggingface.co/)
[![UI: Gradio](https://img.shields.io/badge/UI-Gradio-ff7c00.svg)](https://gradio.app/)
[![Environment: Kaggle](https://img.shields.io/badge/Compute-Kaggle%20T4x2-20BEFF.svg)](https://www.kaggle.com/)

## Overview
This project is an end-to-end NLP pipeline designed to perform stylistic text transformation. It fine-tunes a foundational Large Language Model (**Meta-Llama-3-8B-Instruct**) to translate flowery, AI-generated drafts into highly formal, domain-specific academic prose. The target statistical distribution is tuned on peer-reviewed literature focusing on applied statistics, data science, and financial modeling.

By utilizing a fine-tuned LoRA (Low-Rank Adaptation) adapter, this tool can be deployed via isolated cloud notebooks or self-hosted environments, functioning as a secure, high-privacy editing engine for sensitive or proprietary academic research.

*Note: This project deliberately utilizes **Supervised Fine-Tuning (SFT) via QLoRA** rather than RAG. While RAG is optimal for factual retrieval, it cannot fundamentally alter the statistical burstiness, perplexity, or stylometry of the generator's prose. Fine-tuning was required to securely capture the domain-specific tone.*

---

## The Tech Stack
* **Data Extraction:** Docker, Grobid API
* **Deep Learning Framework:** PyTorch
* **LLM Tooling:** Hugging Face (`transformers`, `peft`, `trl`, `datasets`, `bitsandbytes`)
* **Inference UI:** Gradio
* **Compute Environment:** Kaggle (Dual NVIDIA T4 GPUs)

---

## Architecture & Engineering Workflow

*(Note: Earlier experimental iterations of this pipeline—including standard causal language modeling on unpaired text—have been moved to the `/Archived` directory for historical reference. The current pipeline utilizes a highly optimized supervised translation mapping approach).*

### Phase 1: Data Extraction (Docker & Grobid)
* **Objective:** Extract pure narrative prose from complex academic PDFs while strictly filtering out headers, footers, charts, and reference lists.
* **Process:** PDFs are sent via a Python API to a local `Grobid` machine learning server running inside a Docker container. Grobid parses the layout and returns structured XML (TEI) containing only the abstract and body paragraphs.

### Phase 2: Synthetic Paired Dataset Generation
* **The Problem:** The Llama-3 Instruct model possesses a heavy RLHF "Alignment Tax"—a strong bias toward verbose, conversational "AI-speak" that actively fights custom stylometry.
* **The Solution:** To overwrite this bias, the pipeline utilized a reverse-translation mapping approach. Pure, human-written academic texts were fed into the base LLM to deliberately generate flowery, robotic "AI equivalents." These pairs were then flipped to create a synthetic `[AI-Generated Draft] -> [Human Academic Target]` dataset. This generated ~2,800 perfect training pairs (saved as JSONL), establishing a clear mathematical mapping for stripping AI filler.

### Phase 3: Deep Learning & Fine-Tuning (QLoRA)
* **Objective:** Shift the base Llama-3 weights toward the target academic style using a highly constrained hardware environment (Kaggle NVIDIA T4 GPU, 16GB VRAM).
* **Technique:**
  * The base model is compressed to **4-bit precision (`nf4`)** using `BitsAndBytes`.
  * A **LoRA adapter (Rank=16, Alpha=32)** is attached to the attention modules (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).
  * The model is trained via the `trl` library's `SFTTrainer` (Supervised Fine-Tuning) using exact Chat Template matching.

> **Hardware Troubleshooting:** Llama-3 natively utilizes the `bfloat16` data type. However, free-tier cloud environments (T4 GPUs) physically lack the silicon pathways to compute `bfloat16` gradients natively, resulting in PyTorch AMP GradScaler crashes. To bypass this hardware mismatch without relying on slow software emulation, the `SFTConfig` in this pipeline forces `fp16=False` and `bf16=False`, training the adapter in standard 32-bit math to maintain fast iteration speeds.

### Phase 4: Inference, Gradio UI, & Defeating AI Detectors
* **Deployment:** The final 100MB LoRA adapter is loaded natively over the 8-Billion parameter Meta base model using `PeftModel`. A custom **Gradio Web UI** was built to provide a seamless, side-by-side editing interface.
* **Algorithmic Evasion (Burstiness & Perplexity):** Detectors like Turnitin and GPTZero flag text based on low *burstiness* (sentence length variation) and low *perplexity* (word predictability). Academic writing inherently possesses both, causing frequent false positives. 
* To ensure the generated output passes detection, the inference engine relies on aggressive, custom hyperparameter tuning:
  * `temperature=0.75` and `top_p=0.90` (Nucleus Sampling) to inject controlled mathematical unpredictability.
  * `repetition_penalty=1.12` to prevent repetitive phrasing.
  * Explicit System Prompting to force high burstiness (mixing punchy sentences with complex clauses) while strictly forbidding hallucinated headers or transitions.

---

## Usage

### 1. Model Registry
The trained LoRA adapter weights are hosted privately and can be attached to cloud environments or downloaded for local GGUF conversion.

### 2. Running the Inference UI (Kaggle)
To utilize the editor without requiring local GPU hardware:
1. Open the Kaggle Inference Notebook.
2. Ensure the environment is set to **GPU T4x2** with Internet enabled.
3. Run all cells to initialize the base model, attach the LoRA adapter, and launch the Gradio server.
4. Click the generated `gradio.live` link to open the full-screen Academic Style-Transfer Interface.
5. Paste your drafted text and click "Humanize Text". 

*(For maximum AI detection evasion, it is recommended to apply a 20% manual human pass to the final output to further disrupt the mathematical burstiness pattern).*