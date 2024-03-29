from fastapi import FastAPI
from pydantic import BaseModel
from utils.nlp import extract_ents
from utils.twitter_api import get_response

app = FastAPI()

class Query(BaseModel):
    keyword: str
    max_results: int

@app.get("/")
async def root():
    return {"message": "Hello to Stock Market NLP Analyzer"}

@app.get("/get_tweet_ents")
async def root_post(query:Query):
    return {"query": query}

@app.post("/get_tweet_ents")
async def get_tweet_ents(query:Query):
    data  = get_response(query.keyword,query.max_results)
    data = extract_ents(data)
    return data