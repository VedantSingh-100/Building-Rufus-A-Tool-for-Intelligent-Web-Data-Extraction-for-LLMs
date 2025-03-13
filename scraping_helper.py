import os
import re
import json
from urllib.parse import urlparse, urljoin
import requests
from bs4 import BeautifulSoup
import pdfplumber
from preprocess import clean_text, chunk_text_spacy
import faiss
import numpy as np
# Import the SentenceTransformer and the text generation pipeline.
from sentence_transformers import SentenceTransformer, util, CrossEncoder
from transformers import pipeline
import spacy
nlp = spacy.load("en_core_web_sm")
import torch
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time
import random
import asyncio
import aiohttp
import logging
from urllib.parse import urljoin

if not os.path.exists('scraped_data'):
    os.makedirs('scraped_data')
combined_file = 'scraped_data/combined_content.txt'
preprocessed_file = 'scraped_data/cleaned_content.txt'

relevance_model = SentenceTransformer('all-MiniLM-L6-v2')
USER_QUERY = "Tell me about Carnegie Mellon Museums"
QUERY_EMBEDDING = relevance_model.encode(USER_QUERY, convert_to_tensor=True)

cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# This text2text generation model is used to synthesize an answer.
qa_pipeline = pipeline("text2text-generation", model="t5-base", tokenizer="t5-base")

# Relevance threshold for initial candidate scoring.
RELEVANCE_THRESHOLD = 0.75

def batch_is_link_relevant(anchor_texts, threshold=0.50):
    """
    Processes a list of anchor_texts in batch and returns a list of booleans
    indicating whether each anchor text is relevant.
    """
    # Clean texts and simultaneously filter out generic/short ones.
    cleaned = [clean_text(text) for text in anchor_texts]
    filtered = [(idx, txt) for idx, txt in enumerate(cleaned)]
    
    if not filtered:
        return [False] * len(anchor_texts)
    
    indices, filtered_texts = zip(*filtered)
    
    # Compute embeddings in batch
    embeddings = relevance_model.encode(filtered_texts, convert_to_tensor=True)
    query_embedding = QUERY_EMBEDDING  # Assumed to be precomputed
    
    # Compute cosine similarities using vectorized tensor operations
    cosine_sims = util.pytorch_cos_sim(embeddings, query_embedding)
    # Using squeeze to simplify the shape and then vectorizing the threshold comparison
    relevant_flags = (cosine_sims.squeeze() >= threshold).tolist()
    
    # Reconstruct full results: default to False, then assign computed flags where applicable
    results = [False] * len(anchor_texts)
    for idx, flag in zip(indices, relevant_flags):
        results[idx] = flag
    return results


def remove_unwanted_elements(soup):
    """Removes script and style elements from the parsed HTML."""
    for element in soup(["script", "style"]):
        element.decompose()

def scrape_html(url, filename, force_selenium=False):
    """
    Scrapes an HTML page, extracts paragraphs, writes content to the file, and returns the text.
    If force_selenium is True, skip the requests method and use Selenium directly.
    """
    print(f"Scraping HTML from {url}")

    headers = {
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/90.0.4430.93 Safari/537.36"),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    text_content = None

    # If force_selenium is True, bypass requests
    if not force_selenium:
        try:
            session = requests.Session()
            retry_strategy = Retry(
                total=5,
                backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["HEAD", "GET", "OPTIONS"]
            )
            adapter = HTTPAdapter(max_retries=retry_strategy)
            session.mount("http://", adapter)
            session.mount("https://", adapter)

            time.sleep(random.uniform(1, 3))
            response = session.get(url, headers=headers, timeout=10)
            if response.status_code == 403:
                print(f"Access denied (HTTP 403) when trying to scrape {url}.")
                return None

            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            remove_unwanted_elements(soup)  # Ensure this function is defined
            paragraphs = soup.find_all('p')
            text_content = "\n".join(para.get_text() for para in paragraphs)

            # If content is minimal, consider that it might be dynamically loaded.
            if not text_content.strip() or len(text_content.split()) < 50:
                print(f"Insufficient content from {url} using requests. Falling back to Selenium...")
                raise ValueError("Empty or insufficient content")

            print(f"Content successfully scraped using requests from {url}")

        except Exception as e:
            print(f"Requests scraping failed for {url}: {e}")

    # Selenium fallback (or forced)
    if force_selenium or not text_content or not text_content.strip():
        print("Attempting Selenium fallback...")
        driver = None  # Initialize driver outside the try block
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC

            chrome_options = Options()
            chrome_options.add_argument("--headless=new")  # Updated headless argument
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                                        "Chrome/90.0.4430.93 Safari/537.36")

            driver = webdriver.Chrome(options=chrome_options)
            driver.get(url)

            try:
                # Wait for the main content div to load.  Adjust the timeout as needed.
                WebDriverWait(driver, 15).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "div.main-content"))
                )
            except Exception as wait_e:
                print(f"Timed out waiting for main content to load: {wait_e}")
                driver.quit()  # Ensure driver is closed
                return None # Signal failure

            time.sleep(3)  # Extra delay if needed - could potentially remove if explicit wait is solid

            html = driver.page_source
            soup = BeautifulSoup(html, 'html.parser')
            remove_unwanted_elements(soup)
            paragraphs = soup.find_all('p')
            text_content = "\n".join(para.get_text() for para in paragraphs)

            if text_content.strip():
                print(f"Content successfully scraped using Selenium from {url}")
            else:
                print(f"No paragraph text found for {url} using Selenium.  Check page_source.")
                # Optionally, save driver.page_source to a file for inspection here
                # with open("page_source.html", "w", encoding="utf-8") as f:
                #     f.write(driver.page_source)
                return None

        except Exception as se:
            print(f"Selenium fallback failed for {url}: {se}")
            return None
        finally:
            if driver:
                driver.quit() # Ensure the driver is always closed

    # Write the scraped content to the combined file if content exists.
    if text_content and text_content.strip():
        try:
            with open(filename, 'a', encoding='utf-8') as file:
                file.write(f"\n\nContent from {url}:\n\n")
                file.write(text_content)
            print(f"Data saved for {url}")
        except Exception as file_error:
            print(f"Failed to save data for {url} to {filename}: {file_error}")

    return text_content

def scrape_pdf(pdf_url, file_name):
    """Downloads a PDF, extracts its text, and appends it to the combined file."""
    print(f"Scraping PDF from {pdf_url}")
    try:
        response = requests.get(pdf_url)
        response.raise_for_status()
        pdf_path = f'scraped_data/{file_name}.pdf'
        with open(pdf_path, 'wb') as pdf_file:
            pdf_file.write(response.content)
        with pdfplumber.open(pdf_path) as pdf:
            all_text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    all_text += page_text + "\n"
        if all_text.strip():
            with open(combined_file, 'a', encoding='utf-8') as file:
                file.write(f"\n\nContent from PDF {pdf_url}:\n\n")
                file.write(all_text)
            print(f"PDF text saved for {pdf_url}")
        else:
            print(f"No text extracted from PDF {pdf_url}")
    except Exception as e:
        print(f"Failed to scrape PDF {pdf_url}: {e}")

def filter_links(link, domain):
    """Filters out irrelevant or external links based on common patterns."""
    href = link.get('href', '')
    if any(sub in href for sub in [
        '#cite_note', '#footnote', 'mailto:', 'tel:', 'twitter.com', 'facebook.com',
        'linkedin.com', 'instagram.com', 'youtube.com', 'pinterest.com', '/Site-Footer/',
        'oc_lang=', '?oc_lang=', '/Contact-Us', '/Help:', '/wiki/Help:', '/wiki/File:',
        '/wiki/Category:', '/wiki/Talk:', '/w/index.php', 'javascript:void(0)',
        '#print', '#share', '#cite', '#feedback', 'maps', 'google.com/maps',
        'bing.com/maps', 'apple.com/maps', 'yahoo.com/maps']):
        return False
    if '://' in href and domain not in href:
        return False
    return True

def scrape_subpages(url, domain, level=0, max_depth=1, filename=combined_file, visited=None):
    """Crawls subpages and PDF links (with relevant anchor text) up to a given depth."""
    if visited is None:
        visited = set()
    
    normalized_url = url.rstrip('/')
    if normalized_url in visited:
        return
    visited.add(normalized_url)
    
    if level > max_depth:
        return
    print(f"\nScraping subpages from {url} at level {level}")

    headers = {
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/90.0.4430.93 Safari/537.36"),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    
    session = requests.Session()
    retry_strategy = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD", "OPTIONS"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    try:
        time.sleep(random.uniform(1, 3))
        response = session.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        links = soup.find_all('a', href=True)
        
        anchor_data = []
        for link in links:
            href = link['href']
            if filter_links(link, domain):
                anchor_text = link.get_text(strip=True)
                anchor_data.append((link, anchor_text))
                
        anchor_texts = [data[1] for data in anchor_data]
        relevance_flags = batch_is_link_relevant(anchor_texts, threshold=0.50)
        
        for (link, anchor_text), is_relevant in zip(anchor_data, relevance_flags):
            if is_relevant:
                href = link['href']
                subpage_url = urljoin(url, href).rstrip('/')
                if subpage_url not in visited:
                    # If the URL is known to be dynamic (e.g., sf.gov), force Selenium.
                    force_dynamic = "sf.gov" in subpage_url
                    scrape_html(subpage_url, filename, force_selenium=force_dynamic)
                    scrape_subpages(subpage_url, domain, level=level+1, max_depth=max_depth, filename=filename, visited=visited)
        
        # Process PDF links similarly
        for link in links:
            href = link.get('href', '')
            if href.endswith('.pdf') and filter_links(link, domain):
                anchor_text = link.get_text(strip=True)
                if batch_is_link_relevant(anchor_text):
                    pdf_link = urljoin(url, href).rstrip('/')
                    if pdf_link not in visited:
                        scrape_pdf(pdf_link, filename)
                        visited.add(pdf_link)
                    
    except Exception as e:
        print(f"Failed to scrape subpages or PDFs from {url}: {e}")
