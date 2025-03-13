from fastapi import FastAPI, HTTPException, Query
from pipeline import run_pipeline_with_url

app = FastAPI(title="Scraping and Q&A API")

@app.get("/run_pipeline")
def run_pipeline_endpoint(
    url: str = Query(..., description="URL to scrape"),
    instructions: str = Query("Tell me the FAQ about this company", description="Query instructions")
):
    try:
        result = run_pipeline_with_url(url, instructions)
        return result  # FastAPI will automatically serialize it as JSON.
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))