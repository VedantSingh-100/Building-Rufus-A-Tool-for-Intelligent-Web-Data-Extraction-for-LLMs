# setup.py
from setuptools import setup, find_packages

setup(
    name="Rufus",
    version="0.1.0",
    description="AI-powered selective scraping and Q&A pipeline",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "uvicorn",
        "requests",
        "beautifulsoup4",
        "pdfplumber",
        "sentence-transformers",
        "faiss-cpu",
        "spacy",
        "keybert",
        "transformers",
        "aiohttp",
        "scikit-learn",
    ],
)
