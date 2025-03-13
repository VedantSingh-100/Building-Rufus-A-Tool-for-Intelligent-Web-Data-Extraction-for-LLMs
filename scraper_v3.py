import os
import json
from urllib.parse import urlparse, urljoin
from preprocess import clean_text, chunk_text_spacy
import numpy as np
# Import the SentenceTransformer and the text generation pipeline.
from sentence_transformers import SentenceTransformer, util, CrossEncoder
from transformers import pipeline
import spacy
from sklearn.cluster import KMeans
from keybert import KeyBERT
nlp = spacy.load("en_core_web_sm")
from urllib.parse import urljoin
from scraping_helper import scrape_html, scrape_subpages
from faiss_integration import build_faiss_index, search_faiss_index, cross_encoder_rerank, deduplicate_chunks, synthesize_answer, generate_structured_json_answer
from scraping_functions import preprocess_and_chunk, extract_important_sentences_from_candidates
COMBINED_FILE = 'scraped_data/combined_content.txt'
PREPROCESSED_FILE = 'scraped_data/cleaned_content.txt'
USER_QUERY = "Tell me the FAQ about this company"

# === Model Initialization and Configuration ===
relevance_model = SentenceTransformer('all-MiniLM-L6-v2')
USER_QUERY = "Tell me the FAQ about this company"
QUERY_EMBEDDING = relevance_model.encode(USER_QUERY, convert_to_tensor=True)

cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# This text2text generation model is used to synthesize an answer.
qa_pipeline = pipeline("text2text-generation", model="t5-base", tokenizer="t5-base")

# Relevance threshold for initial candidate scoring.
RELEVANCE_THRESHOLD = 0.75

# This model is used for computing similarity between passages and the query.

# Directory and file for saving scraped content.
if not os.path.exists('scraped_data'):
    os.makedirs('scraped_data')
combined_file = 'scraped_data/combined_content.txt'
preprocessed_file = 'scraped_data/cleaned_content.txt'
QUERY_EMBEDDING = relevance_model.encode(USER_QUERY, convert_to_tensor=True)

def extract_candidate_paragraphs(input_file=preprocessed_file, batch_size=64):
    """
    Reads the preprocessed file (one chunk per line), batch-embeds them,
    computes similarity with QUERY_EMBEDDING, and returns candidate passages.
    """
    if not os.path.exists(input_file):
        print(f"Input file {input_file} not found.")
        return []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        chunks = [line.strip() for line in f.readlines() if line.strip()]
    print(f"Total chunks read: {len(chunks)}")

    # Batch compute embeddings for all chunks.
    embeddings = relevance_model.encode(chunks, convert_to_tensor=True, batch_size=batch_size)
    
    # Compute cosine similarities between each chunk and the query.
    cosine_scores = util.pytorch_cos_sim(embeddings, QUERY_EMBEDDING)  # shape: (n, 1)
    
    candidates = []
    for chunk, score in zip(chunks, cosine_scores):
        candidates.append({
            "url": "N/A",  # URL not preserved; update if needed.
            "paragraph": chunk,
            "score": score.item()
        })
    return candidates

# === Main Pipeline ===

def run_pipeline():
    # --- Step 0: Setup and Initialization ---
    urls = {"sf_gove": "https://simpleorigin.us/"}
    
    # Clear previous data if files are used (or use in-memory strings)
    if os.path.exists(COMBINED_FILE):
        with open(COMBINED_FILE, 'w', encoding='utf-8') as file:
            file.write("")
    
    # --- Step 1: Scraping ---
    for name, url in urls.items():
        scrape_html(url, COMBINED_FILE)
        domain = urlparse(url).netloc
        scrape_subpages(url, domain, level=0, max_depth=1)
    
    # --- Step 2: Preprocess and Chunk ---
    # Option 1: If you refactor preprocess_and_chunk to return chunks
    # chunks = preprocess_and_chunk_content(combined_content)
    # Option 2: If you must work with files, call the function and then read the file.
    if os.path.exists(PREPROCESSED_FILE):
        with open(PREPROCESSED_FILE, 'w', encoding='utf-8') as file:
            file.write("")
    preprocess_and_chunk(input_file=COMBINED_FILE, output_file=PREPROCESSED_FILE)
    
    # Read preprocessed content (assuming one chunk per line)
    with open(PREPROCESSED_FILE, 'r', encoding='utf-8') as f:
        chunks = [line.strip() for line in f.readlines() if line.strip()]
    
    # --- Step 3: Extract Candidate Paragraphs ---
    candidates = extract_candidate_paragraphs(input_file=PREPROCESSED_FILE, batch_size=64)
    if not candidates:
        return {"error": "No candidate paragraphs found."}
    
    candidate_chunks = [c['paragraph'] for c in candidates]
    
    # Compute embeddings and build FAISS index (assuming relevance_model is already initialized)
    embeddings = relevance_model.encode(candidate_chunks, convert_to_tensor=True, batch_size=64)
    index = build_faiss_index(embeddings)
    
    # --- Step 4: Initial FAISS Retrieval ---
    D, I = search_faiss_index(index, QUERY_EMBEDDING, top_k=5)
    faiss_candidates = []
    for score, idx in zip(D[0], I[0]):
        candidate = {
            "url": "N/A",
            "paragraph": candidate_chunks[idx],
            "score": score.item() if hasattr(score, 'item') else score,
        }
        faiss_candidates.append(candidate)
    
    # --- Step 5: Dynamic Label Selection ---
    # Import additional libraries needed for clustering and keyword extraction.
    from sentence_transformers import SentenceTransformer

    import numpy as np
    
    embedding_model_dynamic = SentenceTransformer('all-MiniLM-L6-v2')
    dynamic_embeddings = embedding_model_dynamic.encode(candidate_chunks, show_progress_bar=True)
    num_clusters = 5
    kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(dynamic_embeddings)
    cluster_indices = kmeans.labels_
    
    kw_model = KeyBERT()
    dynamic_labels = []
    for cluster_id in range(num_clusters):
        cluster_texts = [candidate_chunks[i] for i in range(len(candidate_chunks)) if cluster_indices[i] == cluster_id]
        if cluster_texts:
            representative_text = max(cluster_texts, key=len)
            keywords = kw_model.extract_keywords(
                representative_text, keyphrase_ngram_range=(1, 2),
                stop_words='english', top_n=3
            )
            label = keywords[0][0] if keywords else "Unknown"
            dynamic_labels.append(label)
    
    # --- Step 6: Use Dynamic Labels as Subqueries ---
    all_candidates = faiss_candidates.copy()
    # Assume deduplicate_chunks is adapted to work on a list of candidate dictionaries
    all_candidates = deduplicate_chunks(all_candidates)
    
    for label in dynamic_labels:
        # In this example, we use the label directly.
        subquery = f"{USER_QUERY} {label}"
        subq_embedding = relevance_model.encode([subquery], convert_to_tensor=True, batch_size=64)
        D_sub, I_sub = search_faiss_index(index, subq_embedding, top_k=5)
        for score, idx in zip(D_sub[0], I_sub[0]):
            candidate = {
                "url": "N/A",
                "paragraph": candidate_chunks[idx],
                "score": float(score),
            }
            all_candidates.append(candidate)
    
    # --- Step 7: Re-rank Merged Candidates ---
    reranked_candidates = cross_encoder_rerank(USER_QUERY, all_candidates)
    
    # --- Step 8: Extract Important Sentences ---
    important_sentences = extract_important_sentences_from_candidates(
        reranked_candidates, USER_QUERY, cross_encoder, nlp, top_k=5
    )
    
    # --- Step 9: Synthesize Answer ---
    answer, top_passages = synthesize_answer(USER_QUERY, important_sentences, top_k=5)

    result = generate_structured_json_answer(USER_QUERY, answer, top_passages)
    return result


if __name__ == "__main__":
    # For testing purposes, print the result.
    result = run_pipeline()
    print(json.dumps(result, indent=4))
