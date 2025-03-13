# Rufus/client.py
import os
from pipeline import run_pipeline_with_url  # Assuming you refactor your pipeline to accept a URL parameter

class RufusClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        # You can use the API key to set up credentials or configuration as needed.

    def scrape(self, url: str, instructions: str = None) -> dict:
        """
        Runs the scraping and processing pipeline on the given URL.
        
        :param url: URL to scrape.
        :param instructions: Optional instructions for processing.
        :return: A structured JSON response containing the query, generated answer, and supporting passages.
        """
        # Optionally, you can use 'instructions' in your pipeline if needed.
        result = run_pipeline_with_url(url, instructions=instructions)
        return result
