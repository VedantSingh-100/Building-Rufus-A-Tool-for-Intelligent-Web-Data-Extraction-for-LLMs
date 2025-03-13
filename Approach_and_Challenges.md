# Project Documentation: AI-Powered Selective Scraping & Q&A Pipeline

## Project Summary

This project implements an AI-powered pipeline designed to perform selective web scraping, data cleaning, intelligent candidate extraction, and answer synthesis based on user queries. The pipeline processes content from target websites and generates coherent answers using advanced NLP models and deep learning techniques. The entire workflow is exposed via a REST API built with FastAPI.

## Approach

1. **Selective Web Scraping by an AI Agent:**  
   - An intelligent agent guides the scraping process to selectively follow the most relevant links on a website.
   - This ensures that only high-quality and contextually relevant content is retrieved, reducing noise and unnecessary data.

2. **AI-Driven Data Cleaning & Preprocessing:**  
   - Raw web content is cleaned using NLP techniques to remove noise and standardize the text.
   - The cleaned data is then chunked into manageable segments to enable efficient downstream processing.

3. **Candidate Extraction & Retrieval:**  
   - Text embeddings are computed for each content chunk using models from the SentenceTransformer library.
   - FAISS is used for efficient similarity searches, allowing the system to quickly identify candidate passages that match the user query.
   - Clustering (using KMeans) and keyword extraction (via KeyBERT) dynamically label content for improved relevance during retrieval.

4. **Re-ranking & Answer Synthesis:**  
   - A cross-encoder re-ranks candidate passages based on their contextual relevance to the query.
   - A text generation model synthesizes a coherent answer by integrating the highest-ranked passages.
  
5. **API Integration:**  
   - The entire process is wrapped in a FastAPI application.
   - The `/run_pipeline` endpoint triggers the pipeline and returns a structured JSON response containing the query, generated answer, and supporting passages.

## Challenges and Solutions

- **Selective Scraping:**  
  *Challenge:* Identifying and following only the most relevant links among a vast number of available URLs.  
  *Solution:* Implemented an AI agent that intelligently determines which links to follow based on contextual relevance, significantly reducing noise and improving data quality.

- **Data Cleaning & Preprocessing:**  
  *Challenge:* Handling the diverse and unstructured nature of web data.  
  *Solution:* Developed robust text cleaning and chunking functions using advanced NLP tools (such as spaCy and custom preprocessing functions) to standardize and preprocess the content effectively.

- **Data Serialization:**  
  *Challenge:* Encountering JSON serialization issues due to non-native Python data types (e.g., `numpy.float32`).  
  *Solution:* Converted all non-serializable types to native Python types before generating the JSON output, ensuring seamless serialization.

- **Integration of Multiple Models and Libraries:**  
  *Challenge:* Combining various advanced libraries (FAISS, SentenceTransformer, KeyBERT, FastAPI, etc.) into a cohesive pipeline.  
  *Solution:* Modularized the code by encapsulating distinct functionalities into well-defined functions, making integration smoother and debugging easier.

- **Performance and Scalability:**  
  *Challenge:* Ensuring the pipeline processes data efficiently despite the heavy computational requirements of web scraping and model inference.  
  *Solution:* Utilized asynchronous processing and batching techniques where applicable, optimizing the workflow and reducing latency.

## Conclusion

This project showcases a comprehensive approach to building an AI-driven data processing pipeline. By integrating selective web scraping, advanced data cleaning, efficient candidate retrieval, and dynamic answer synthesis, the pipeline delivers precise and contextually relevant answers via a user-friendly API. The careful modularization of tasks and handling of performance challenges have resulted in a robust, scalable system that leverages the best of modern AI and NLP technologies.

