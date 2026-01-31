# Graph-Based RAG Fake News Detection System

A Retrieval-Augmented Generation (RAG) pipeline for **fake news detection and claim verification**, enhanced with **graph-based evidence clustering** and **multi-hop retrieval**. This project grounds Large Language Model (LLM) outputs in verifiable evidence to reduce hallucinations and improve transparency.

---

## 🚀 Overview

Large Language Models are powerful but prone to hallucinations when answering factual questions. This project addresses that limitation by combining:

- **Verified evidence retrieval** (RAG)
- **Graph-based clustering of evidence** (k-means++)
- **Multi-hop semantic search**
- **LLM-based fact verification with justification**

Given an input claim, the system retrieves relevant evidence from a structured knowledge base, ranks it for relevance, and produces a **fact-checked decision** (e.g., *Supported* or *Refuted*) along with an **evidence-backed explanation**.

---

## 🧠 Key Ideas

- Claims and evidence are embedded into a shared semantic space using **Sentence Transformers**
- Evidence embeddings are compressed using **PCA** and clustered into semantic groups
- Clusters form a **graph-like structure** for efficient and focused retrieval
- A **cross-encoder** re-ranks candidate evidence for precision
- An **instruction-following LLM (FLAN-T5)** performs final claim verification

This design improves reliability, recall, and interpretability compared to vanilla LLM approaches.

---

## 🏗️ System Architecture

```
User Claim
   ↓
Embedding (Sentence Transformer)
   ↓
Relevant Cluster Selection (k-means++ centroids)
   ↓
Multi-Hop Evidence Retrieval
   ↓
Cross-Encoder Re-ranking
   ↓
LLM Fact Verification + Justification
```

---

## 📁 Project Structure

```bash
.
├── claim_extractor.py           # Extracts atomic, verifiable claims from news articles
├── guardian_fetcher.py          # Fetches news articles from credible sources
├── generate_RAG_database.py     # Builds the evidence database (embeddings + clusters)
├── fake_news_detector_rag.py    # End-to-end RAG pipeline for claim verification
├── fact_checker_llm.py          # LLM-based fact checking and justification generation
├── Deep Learning Final Report.pdf
├── Implementing RAG Graphs to Enhance LLM Capability.pdf
└── README.md
```

---

## 🔍 Pipeline Breakdown

### 1. Data Collection & Claim Extraction

- News articles are fetched from credible sources (2025 time range)
- Articles are decomposed into **short, atomic factual claims** using **FLAN-T5-base**

### 2. Embedding & Preprocessing

- Claims and evidence are embedded using `sentence-transformers/all-MiniLM-L6-v2`
- Embeddings are reduced from 768 → ~300 dimensions using **PCA**
- Vectors are **L2-normalized** for efficient cosine similarity search

### 3. Graph-Based Clustering

- Evidence embeddings are grouped using **k-means++ clustering**
- Each cluster represents a semantically coherent topic
- Cluster centroids act as graph nodes for retrieval

### 4. Multi-Hop Retrieval & Re-Ranking

- Top-X relevant clusters are selected per query
- Top-N evidence candidates are retrieved per hop
- A **cross-encoder** (`ms-marco-MiniLM-L6-v2`) re-ranks candidates
- Multi-hop retrieval allows exploration beyond initial matches

### 5. Claim Verification

- The input claim + top-K evidence are passed to **FLAN-T5-base**
- Output includes:
  - Claim classification (Supported / Refuted)
  - Natural-language justification grounded in evidence

---

## ⚙️ Configuration & Hyperparameters

| Parameter | Value |
|--------|-------|
| NUM_CLUSTERS | 17 |
| TOP_X (clusters) | 3 |
| TOP_N (candidates per hop) | 10 |
| TOP_K (final evidence) | 4 |
| NUM_HOPS | 2 |
| PCA Variance Retained | 90% |
| Similarity Cutoff | 0.8 |

---

## 📊 Evaluation

- Evaluated on a **balanced FEVER dataset**
- 300 supported claims / 300 refuted claims
- Metrics used:
  - Precision
  - Recall
  - F1-score

### Key Result

✅ **~23% performance improvement** compared to an LLM without RAG, even with minimal training data.

High recall was observed due to the model’s conservative design (assumes false unless evidence supports the claim).

---

## ⚠️ Known Challenges

- Limited coverage of the knowledge base for unseen topics
- Suboptimal clustering can dilute relevant evidence
- Multi-hop retrieval quality depends heavily on evidence density

---

## 🔮 Future Work

- Expand dataset with broader and more diverse verified sources
- Experiment with alternative graph representations (explicit edges, GNNs)
- Try clustering alternatives beyond k-means++
- Improve evidence chunking and semantic granularity
- Enhance LLM grounding and explanation quality

---

## 🧑‍💻 Authors

- **Arsam Ahmad**
- Matthew Shoup
- Mridul Madan

---

## 📚 References

- FEVER: Fact Extraction and VERification Dataset
- Yang et al. (2018) – TI-CNN
- Nasir et al. (2021) – CNN–RNN Fake News Detection
- Chang et al. (2024) – Graph Global Attention Networks
- Lewis et al. (2020) – Retrieval-Augmented Generation

---

## ⭐ Final Note

This project demonstrates how **structured retrieval + graph reasoning** can significantly improve the trustworthiness of LLM-based systems. It is designed as both a **research prototype** and a **practical foundation** for real-world fact-checking systems.

If you’re exploring RAG, fake news detection, or trustworthy AI — this is a great place to start.

