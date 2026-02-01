# Custom Paraphrase Generator (CPG) - Comprehensive Analysis

## 📋 Project Overview

This project implements a **Custom Paraphrase Generator (CPG)** that compares multiple state-of-the-art approaches for paraphrasing long-form text while maintaining semantic meaning and adhering to a minimum length requirement of 80% of the original text length.

The notebook evaluates five distinct paraphrasing strategies, ranging from lightweight transformer models to advanced large language models, providing a comprehensive comparison of their effectiveness, efficiency, and output quality.

---

## 🎯 Objectives

- **Primary Goal**: Develop robust paraphrasing solutions for long-form documents (cover letters, essays, articles)
- **Secondary Goals**:
  - Compare multiple architectural approaches
  - Ensure semantic preservation while achieving lexical diversity
  - Maintain minimum length constraints (80% of original)
  - Evaluate output quality using both automated metrics and LLM-based judgment

---

## 🏗️ Architectures Evaluated

### 1. **T5-small** (Lightweight Baseline)
- **Type**: Transformer-based Seq2Seq model
- **Pros**: Fast inference, low memory footprint
- **Cons**: Struggles with long-context text, loses semantic meaning toward the end
- **Use Case**: Quick baseline for short texts

### 2. **T5-base** (Enhanced Baseline)
- **Type**: Transformer-based Seq2Seq model (larger variant)
- **Pros**: Better contextual understanding than T5-small
- **Cons**: Token truncation issues with 340+ word documents
- **Use Case**: Medium-length documents

### 3. **T5-base (Chunked Strategy)**
- **Type**: T5-base with semantic-aware chunking
- **Pros**: Preserves semantic boundaries, maintains coherence across chunks
- **Cons**: Slightly increased latency due to multiple passes
- **Use Case**: Long-form documents requiring coherence preservation
- **Implementation**: Sentence-level chunking with SentenceTransformer embeddings

### 4. **T5-base (Optimized Parameters)**
- **Type**: T5-base with fine-tuned hyperparameters
- **Pros**: Best transformer-based approach, balanced quality and speed
- **Cons**: Still limited by transformer token window
- **Use Case**: Production deployment with quality guarantees
- **Key Optimizations**:
  - Adjusted beam search strategy (`num_beams=4`)
  - Conservative temperature settings (`temperature=1.0`)
  - Proper length penalty tuning (`length_penalty=1.0`)
  - Early stopping enabled for efficiency

### 5. **Ollama LLM (llama3.1:8b)** (SOTA)
- **Type**: Large language model with instruction-following capabilities
- **Pros**: Superior semantic preservation, most human-like output, maintains originality
- **Cons**: Requires local Ollama setup, slower inference
- **Use Case**: High-quality paraphrasing where speed is secondary
- **Implementation**: System prompt-based approach for task-specific behavior

---

## 📊 Methodology

### Why T5 Architecture?

T5 (Text-to-Text Transfer Transformer) was selected as the primary transformer-based solution for the following reasons:

```
┌─────────────────────────────────────────────────────┐
│         T5 vs GPT vs BERT Comparison                │
├─────────────────────────────────────────────────────┤
│ Aspect           │ GPT   │ BERT  │ T5              │
├──────────────────┼───────┼───────┼─────────────────┤
│ Architecture     │ Decoder│Encoder│Encoder-Decoder│
│ Masking          │ Causal│Bidirectional│Full     │
│ Task Framework   │Generation│Classification│Text2Text│
│ Paraphrase Fit   │ ⚠️    │ ❌    │ ✅ (Perfect)  │
└─────────────────────────────────────────────────────┘
```

**Key Advantages**:
1. **Sequence-to-Sequence Design**: Paraphrasing is inherently a "text-in, text-out" task
2. **Prefix-based Triggering**: Simple `paraphrase:` prefix activates task-specific generation
3. **Length Control**: `min_length` and `max_length` parameters enable 80% requirement enforcement
4. **Full Transformer Architecture**: Captures complex linguistic patterns with both encoding and decoding

**Optimization Contributions from Copilot**:
- Proposed token-count-based splitting (instead of character-based)
- Suggested SentenceTransformer embeddings for semantic similarity detection
- Recommended optimal chunk size (300 tokens) for balance between coherence and efficiency

### LLM as Judge Evaluation

A sophisticated evaluation framework uses the LLM itself as a quality judge:

```
Evaluation Dimensions:
├─ Semantic Preservation (1-10)
├─ Lexical Diversity (1-10)
├─ Fluency (1-10)
├─ Coherence (1-10)
├─ Originality (1-10)
└─ Overall Score (Average)
```

---

## 📈 Evaluation Metrics

### Automated Metrics

| Metric | Purpose | Formula |
|--------|---------|---------|
| **BLEU Score** | N-gram overlap with original | Geometric mean of n-gram precisions |
| **ROUGE Score** | Recall-based evaluation | ROUGE-1 (unigram), ROUGE-L (longest common subsequence) |
| **Semantic Similarity** | Meaning preservation | TF-IDF + Cosine Similarity |
| **Length Ratio** | Constraint compliance | Paraphrased Length / Original Length |

### LLM-Based Judgment

- **Multi-dimensional scoring** across 5 key dimensions
- **Qualitative reasoning** for each score
- **Overall assessment** with human-readable summary
- **Reproducible evaluation** with temperature=0 for consistency

---

## 🛠️ Technical Stack

```
Core Libraries:
├─ transformers (HuggingFace) - Model loading and generation
├─ sentence-transformers - Semantic embeddings for chunking
├─ ollama - LLM inference and evaluation
├─ nltk - BLEU score calculation
├─ rouge_score - ROUGE metric computation
├─ scikit-learn - TF-IDF and cosine similarity
└─ pandas - Results aggregation and display

Models:
├─ t5-small (60M parameters)
├─ t5-base (220M parameters)
└─ llama3.1:8b (8B parameters, via Ollama)

Evaluation Frameworks:
├─ BLEU - N-gram based metrics
├─ ROUGE - Recall-based metrics
├─ Semantic Similarity - TF-IDF + Cosine
└─ LLM-as-Judge - Multi-dimensional assessment
```

---

## 🤖 VS Code Copilot Integration

This project leveraged **VS Code Copilot** for optimization and development assistance:

### Areas Where Copilot Provided Value

1. **Chunking Algorithm Optimization**
   - Suggested token-aware splitting logic
   - Optimized boundary detection
   - Recommended SentenceTransformer integration

2. **Code Quality & Efficiency**
   - Improved error handling patterns
   - Suggested vectorized operations
   - Optimized memory management in batch processing

---
