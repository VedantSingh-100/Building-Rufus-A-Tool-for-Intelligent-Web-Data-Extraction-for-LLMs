import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, util, CrossEncoder
from transformers import pipeline
import torch
import json
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
qa_pipeline = pipeline("text2text-generation", model="t5-base", tokenizer="t5-base")

def build_faiss_index(embeddings):
    """
    Builds a FAISS index using inner-product (IP) which is equivalent to cosine similarity
    if the embeddings are L2 normalized.
    """
    emb_np = embeddings.cpu().numpy().astype('float32')
    dimension = emb_np.shape[1]
    index = faiss.IndexFlatIP(dimension)
    # Normalize embeddings to unit length
    faiss.normalize_L2(emb_np)
    index.add(emb_np)
    return index

def search_faiss_index(index, query_embedding, top_k=5):
    """
    Searches the FAISS index for the top_k most similar embeddings.
    Ensures query_embedding is 2D.
    """
    query_np = query_embedding.cpu().numpy().astype('float32')
    # If the query is 1D, expand it to 2D
    if query_np.ndim == 1:
        query_np = np.expand_dims(query_np, axis=0)
    faiss.normalize_L2(query_np)
    D, I = index.search(query_np, top_k)
    return D, I

def cross_encoder_rerank(query, candidate_passages):
    """
    Uses a cross-encoder to re-rank candidate passages.
    Returns a list of candidate passages with updated scores.
    """
    # Create a list of (query, passage) pairs
    cross_inputs = [(query, passage["paragraph"]) for passage in candidate_passages]
    cross_scores = cross_encoder.predict(cross_inputs)
    
    # Update each candidate's score with the cross-encoder score
    for idx, score in enumerate(cross_scores):
        candidate_passages[idx]["score"] = score
    # Re-rank by updated score (descending)
    candidate_passages.sort(key=lambda x: x["score"], reverse=True)
    return candidate_passages

def generate_answer(query, context):
    """
    Uses a text-to-text generation model (T5) to generate an answer based on the given query and context.
    """
    prompt = f"question: {query} context: {context}"
    output = qa_pipeline(prompt, max_length=150, truncation=True)
    answer = output[0]['generated_text']
    return answer

def synthesize_answer(query, candidate_paragraphs, top_k=5):
    """
    Sorts candidate paragraphs by their similarity score, selects the top-k passages,
    concatenates them as context, and then uses a text generation model to synthesize an answer.
    Returns the answer along with the supporting passages.
    """
    # Sort by descending similarity score.
    candidate_paragraphs.sort(key=lambda x: x["score"], reverse=True)
    top_candidates = candidate_paragraphs[:top_k]
    context = "\n".join([cand["sentence"] for cand in top_candidates])
    answer = generate_answer(query, context)
    return answer, top_candidates

def generate_structured_json_answer(query, answer, supporting_passages):
    """
    Generates a structured JSON document (as a dictionary) that contains the user query,
    the generated answer, and a list of supporting passages (with their source URLs and scores).
    """
    # Convert all scores to Python's float type.
    for passage in supporting_passages:
        if isinstance(passage.get("score"), (np.float32, np.float64)):
            passage["score"] = float(passage["score"])
    
    data = {
        "query": query,
        "answer": answer,
        "supporting_passages": supporting_passages
    }
    
    return data

from sentence_transformers import SentenceTransformer, util

def deduplicate_chunks(chunks, similarity_threshold=0.95):
    """Remove semantically similar chunks using SBERT"""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(chunks, convert_to_tensor=True)
    cos_sim = util.cos_sim(embeddings, embeddings)
    
    unique_indices = []
    seen = set()
    
    for i in range(len(chunks)):
        if i not in seen:
            unique_indices.append(i)
            # Mark similar chunks as seen
            similar = torch.where(cos_sim[i] > similarity_threshold)[0]
            seen.update(similar.tolist())
    
    return [chunks[i] for i in unique_indices]