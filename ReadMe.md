# AI-Powered Selective Scraping & Q&A Pipeline

## Overview

This project implements an advanced, AI-powered pipeline that:
- **Selectively scrapes** websites using an AI agent that intelligently identifies and follows the most relevant links.
- **Cleans and preprocesses** web data using state-of-the-art natural language processing (NLP) techniques.
- **Extracts, ranks, and clusters** candidate content using deep learning models, FAISS for similarity search, and clustering with KMeans.
- **Synthesizes answers** based on a user query by leveraging text generation models.
- **Exposes a REST API** via FastAPI to trigger the entire process and return a structured JSON output.

## Key Features

- **Selective Scraping by an AI Agent:**  
  Uses an intelligent agent to navigate and select only the most relevant links from a website, ensuring that your pipeline processes high-quality and pertinent content.

- **AI-Driven Data Cleaning:**  
  Incorporates advanced NLP models to clean and preprocess raw web data, removing noise and ensuring that only useful textual content is passed to the subsequent stages.

- **Efficient Content Retrieval:**  
  Leverages FAISS and SentenceTransformer to quickly compute embeddings and perform similarity searches across large text datasets.

- **Dynamic Content Analysis:**  
  Uses clustering techniques (KMeans) and keyword extraction (KeyBERT) to dynamically label and segment content, enabling more precise retrieval and processing of candidate passages.

- **Cross-Encoder Re-ranking:**  
  Re-ranks candidate passages using a cross-encoder model to ensure that the most contextually relevant information is used for generating answers.

- **Answer Synthesis:**  
  Employs a text-to-text generation model (e.g., T5) to synthesize a coherent answer from the selected passages.

- **REST API with FastAPI:**  
  Provides an easy-to-use API endpoint (`/run_pipeline`) that executes the entire pipeline and returns a structured JSON document containing the query, generated answer, and supporting passages.

## Requirements

- Python 3.8+
- Dependencies are listed in `requirements.txt`:
  - fastapi
  - uvicorn
  - requests
  - beautifulsoup4
  - pdfplumber
  - sentence-transformers
  - faiss-cpu
  - spacy
  - keybert
  - transformers
  - aiohttp
  - scikit-learn
  - (plus any other packages required by the pipeline)

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd <repository_directory>
