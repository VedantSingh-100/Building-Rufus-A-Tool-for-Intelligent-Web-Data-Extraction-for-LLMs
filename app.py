from fastapi import FastAPI, HTTPException
from scraper_v3 import run_pipeline

app = FastAPI(title="Scraping and Q&A API")

@app.get("/run_pipeline")
def run_pipeline_endpoint():
    try:
        result = run_pipeline()
        return result  # FastAPI will automatically serialize it as JSON.
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))