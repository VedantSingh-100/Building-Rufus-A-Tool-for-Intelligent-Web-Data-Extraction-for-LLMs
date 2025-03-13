import os
import re
import spacy
from bs4 import BeautifulSoup

from spacy.lang.en import English 

nlp = English()

nlp.max_length = 1000000  

nlp.add_pipe("sentencizer")

#Directory containing scraped text files
scraped_data_dir = 'scraped_data'
combined_file = 'C:/Users/Amulya/Documents/CMU_Sem3/ANLP/RAG-QA-LLM-Pipeline/scraped_data/combined_content5.txt'
preprocessed_file = 'C:/Users/Amulya/Documents/CMU_Sem3/ANLP/RAG-QA-LLM-Pipeline/scraped_data/preprocessed_content5.txt'

def clean_text(text):
    text = text.lower()

    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    text = re.sub(r'\[\d+\]', '', text)

    #text = re.sub(r'[@#$%^&*()_+\-=\[\]{}|<>]', '', text)

    #text = re.sub(r'–|—', '-', text)  # Replace em-dashes with hyphens

    text = re.sub(r'\d+\.\s+', '', text)  # Remove "1. ", "2. ", etc.
    text = re.sub(r'[•●▪]', '', text)  # Remove bullet points

    text = re.sub(r'\.{2,}', '.', text)

    text = re.sub(r'(section|article)\s?\d+[-:]?', '', text)

    text = re.sub(r'\s+', ' ', text).strip()

    return text

def chunk_text_spacy(text, max_chunk_size=5):
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents] 

    chunks = [' '.join(sentences[i:i + max_chunk_size]) for i in range(0, len(sentences), max_chunk_size)]

    return chunks

def preprocess_and_chunk():
    try:
        with open(combined_file, 'r', encoding='utf-8') as file:
            content = file.read()

        cleaned_content = clean_text(content)

        chunks = chunk_text_spacy(cleaned_content, max_chunk_size=5) # changed to 1, 2, 5, 7, 10

        with open(preprocessed_file, 'w', encoding='utf-8') as file:
            for chunk in chunks:
                file.write(chunk + '\n')
        print(f"Preprocessed and chunked data saved to {preprocessed_file}")

    except Exception as e:
        print(f"Failed to preprocess and chunk data: {e}")

if __name__ == "__main__":
    preprocess_and_chunk()
