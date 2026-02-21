# Richo_GPT_Data_Dons
Hybrid RAG Retrieval Pipeline for Ricoh Technical Documentation


# Ricoh Technical Documentation RAG System

A Retrieval-Augmented Generation (RAG) pipeline built to answer technical questions about Ricoh printing systems. It loads PDF documentation, indexes it using both semantic and keyword-based search, and uses a large language model to synthesize accurate, cited answers from the retrieved content.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [How It Works](#how-it-works)
- [Configuration](#configuration)
- [Retrieval Strategy](#retrieval-strategy)
- [The Agent](#the-agent)
- [Known Limitations](#known-limitations)

---

## Overview

This system is designed to help users find answers within large collections of Ricoh PDF manuals and technical documents without manually searching through them. Instead of relying on simple keyword matching or a single embedding model, it combines two different search strategies and fuses their results for better accuracy.

The pipeline has two stages. The first stage is indexing, where documents are processed and stored in a way that makes them fast to search. The second stage is querying, where a user asks a question, the system finds relevant document chunks, and an LLM generates a structured answer based only on what was found.

---

## Architecture

```
PDF Documents (folder)
        |
        v
  PDFMinerLoader
  (loads each PDF page by page)
        |
        v
RecursiveCharacterTextSplitter
  (breaks pages into 500-char chunks
   with 50-char overlap)
        |
        v
   ┌────┴────┐
   |         |
   v         v
 FAISS     BM25Okapi
(semantic) (keyword)
   |         |
   └────┬────┘
        |
        v
 Reciprocal Rank Fusion
 (combines both result sets
  into a single ranked list)
        |
        v
  Retrieved Chunks
  (top-k most relevant)
        |
        v
   Groq LLaMA 3.3 70B
   (LangChain Agent)
        |
        v
  Structured JSON Response
  (topic, summary, sources)
```

---

## Project Structure

```
project/
├── RPD-en-US/             # Folder containing all Ricoh PDF documentation
├── faiss_index/           # Saved FAISS vector index (generated on first run)
├── bm25.pkl               # Saved BM25 keyword index (generated on first run)
├── retriever.pkl          # Saved document chunks (generated on first run)
├── data_pdf_parse.py      # Main pipeline script
├── .env                   # Environment variables (GROQ_API_KEY)
└── README.md
```

---

## Prerequisites

- Python 3.9 or above
- A Groq API key (free tier available at console.groq.com)
- PDF documents placed inside a folder named `RPD-en-US`

---

## Installation

Clone or download the project, then install all dependencies:

```bash
pip install langchain langchain-core langchain-community langchain-huggingface
pip install langchain-groq langchain-text-splitters
pip install faiss-cpu sentence-transformers
pip install rank_bm25 pdfminer.six python-dotenv
```

Create a `.env` file in the root of the project and add your Groq API key:

```
GROQ_API_KEY=your_api_key_here
```

Place all your Ricoh PDF files inside the `RPD-en-US` folder. The script will automatically pick up every `.pdf` file in that directory.

---

## How to Run

Simply run the main script:

```bash
python data_pdf_parse.py
```

On the first run, the script will:
1. Load and parse all PDFs in the `RPD-en-US` folder
2. Split them into chunks
3. Build and save the FAISS and BM25 indexes to disk

Once indexing is done, it will prompt you to enter a question:

```
Enter Your Query: ```

Type your question and press Enter. The agent will retrieve relevant chunks from the documentation and return a structured response with a topic, summary, and cited sources along with PDF on the right side.

---

## Working: 

### Step 1 - Document Loading

`PDFMinerLoader` reads each PDF file page by page. Setting `concatenate_pages=False` keeps individual pages as separate documents, which preserves page number metadata used later in citations.

### Step 2 - Chunking

`RecursiveCharacterTextSplitter` breaks each page into smaller chunks of 500 characters with a 50-character overlap between consecutive chunks. The overlap ensures that context is not lost at chunk boundaries. For example, if a sentence spans the end of one chunk and the beginning of the next, both chunks carry enough surrounding text to make sense independently.

### Step 3 - Indexing

Two separate indexes are built from the chunks:

**FAISS** converts each chunk into a dense vector using the `all-MiniLM-L6-v2` sentence transformer model. This model maps semantically similar text to nearby points in vector space, so a query like "how to reset the fuser unit" can match a chunk that talks about "replacing thermal components" even if no exact words overlap.

**BM25** builds a sparse keyword index using term frequency statistics. It is better at exact matches, product codes, error codes, or any situation where the user knows the specific terminology used in the manual.

Both indexes are saved to disk after the first run, so subsequent runs skip the indexing step entirely and load from the saved files.

### Step 4 - Hybrid Search with Reciprocal Rank Fusion

When a query comes in, it is sent to both FAISS and BM25 independently. Each returns a ranked list of document chunks. These two lists are then merged using Reciprocal Rank Fusion (RRF).

RRF works by assigning a score to each chunk based on its rank position in each list, using the formula `1 / (60 + rank)`. Chunks that appear near the top of both lists receive the highest combined scores. The `alpha` parameter controls how much weight each retriever gets. The default is `0.5` for equal weighting.

### Step 5 - Answer Generation

The top retrieved chunks are passed to a LangChain agent backed by Groq's LLaMA 3.3 70B model. The agent is instructed to only use information from the provided evidence, cite every major claim inline using the format `[Doc: X, Page: Y, Section: Z]`, and structure its answer with numbered steps where appropriate. If the evidence does not contain enough information to answer the question, the agent is instructed to say so rather than guess.

The response is returned as a structured JSON object with three fields: `topic`, `summary`, and `sources`.

---

## Configuration

The following values can be adjusted at the top of the script:

| Parameter | Default | Description |
|---|---|---|
| `folder_path` | `"RPD-en-US"` | Folder containing PDF files |
| `chunk_size` | `500` | Maximum character length per chunk |
| `chunk_overlap` | `50` | Characters shared between adjacent chunks |
| `top_k` | `3` | Number of results returned by hybrid search |
| `alpha` | `0.5` | Weight for FAISS vs BM25 (0 = full BM25, 1 = full FAISS) |
| `model` | `"llama-3.3-70b-versatile"` | Groq model used for answer generation |

---

## Retrieval Strategy

The decision to use hybrid search rather than pure semantic search comes down to the nature of technical documentation. Ricoh manuals contain a lot of precise terminology, model numbers, error codes, and step-by-step procedures. A purely semantic model might retrieve chunks that are topically related but miss the specific model or error code the user is asking about. BM25 fills that gap by rewarding exact term matches.

Conversely, users do not always phrase questions using the exact language from the manual. A pure BM25 approach would fail when the user paraphrases or uses different terminology. FAISS handles those cases by finding chunks that are semantically equivalent even when the wording differs.

Combining both with RRF produces results that are both semantically relevant and terminologically precise.

---

## The Agent

The LangChain agent wraps the `hybrid_search` function as a tool called `search_ricoh_docs`. When the agent receives a query, it decides when and how many times to call this tool based on the complexity of the question. The `max_iterations=3` setting limits how many tool calls the agent can make before it must produce a final answer, which keeps latency reasonable.

The system prompt instructs the agent to behave as a technical support specialist, focus on hardware specs, software configuration, operating procedures, and compatibility, and never add information that is not present in the retrieved evidence.

---

## Known Limitations

- The first run is slow because it must process all PDFs and build both indexes. For a large document collection this can take several minutes.
- PDFs with scanned images rather than embedded text will not be parsed correctly by PDFMiner. If your documents are image-based, you would need to add an OCR step before loading.
- The `chunk_size=500` setting works well for most documentation but may produce incomplete chunks for documents with very long tables or code blocks.
- The agent is limited to information contained within the indexed documents. Questions about topics not covered in the PDFs will return an acknowledgment that the evidence is insufficient rather than a hallucinated answer.
