
combined_file = 'scraped_data/combined_content.txt'
preprocessed_file = 'scraped_data/cleaned_content.txt'
from preprocess import clean_text
import numpy as np
import os
from faiss_integration import deduplicate_chunks
from sentence_transformers import SentenceTransformer, util, CrossEncoder
relevance_model = SentenceTransformer('all-MiniLM-L6-v2')

# === Scraping Functions ===

def improved_chunking(text, min_words=50, max_words=200):
    """Chunk while preserving paragraph context"""
    chunks = []
    current_chunk = []
    current_word_count = 0
    
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]  # Use double newlines
    
    for para in paragraphs:
        words = para.split()
        para_word_count = len(words)
        
        if para_word_count > max_words:
            # Split large paragraphs into sub-chunks
            for i in range(0, para_word_count, max_words):
                chunk = ' '.join(words[i:i+max_words])
                chunks.append(chunk)
        elif current_word_count + para_word_count <= max_words:
            # Build multi-paragraph chunks
            current_chunk.append(para)
            current_word_count += para_word_count
        else:
            # Finalize current chunk
            chunks.append('\n'.join(current_chunk))
            current_chunk = [para]
            current_word_count = para_word_count
    
    if current_chunk:
        chunks.append('\n'.join(current_chunk))
    
    return chunks


def preprocess_and_chunk(input_file, output_file):
    try:
        with open(input_file, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Clean before chunking
        cleaned = clean_text(content)
        
        # Semantic chunking
        chunks = improved_chunking(cleaned)
        
        # Semantic deduplication
        unique_chunks = deduplicate_chunks(chunks)
        
        with open(output_file, 'w', encoding='utf-8') as file:
            file.write('\n'.join(unique_chunks))
            
    except Exception as e:
        print(f"Preprocessing failed: {e}")


def extract_top_sentence(chunk, query, cross_encoder, nlp):
    """
    Splits a candidate chunk into sentences, then uses the cross-encoder
    to select the most relevant sentence to the query.
    """
    doc = nlp(chunk)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    if not sentences:
        return None, 0.0
    # Prepare (query, sentence) pairs for the cross-encoder.
    input_pairs = [(query, sentence) for sentence in sentences]
    scores = cross_encoder.predict(input_pairs)
    best_idx = np.argmax(scores)
    return sentences[best_idx], scores[best_idx]

def extract_important_sentences_from_candidates(candidate_passages, query, cross_encoder, nlp, top_k=5):
    """
    For each candidate passage, extract its most relevant sentence.
    Returns the top_k sentences (with their scores) across all candidates.
    """
    results = []
    for candidate in candidate_passages:
        chunk = candidate["paragraph"]
        best_sentence, score = extract_top_sentence(chunk, query, cross_encoder, nlp)
        if best_sentence:
            results.append({
                "url": candidate["url"],
                "sentence": best_sentence,
                "score": score
            })
    # Sort by score in descending order.
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]