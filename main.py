# -*- coding: utf-8 -*-
"""
Created on Thu Aug  7 19:17:25 2025

@author: advit
"""

from fastapi import FastAPI
import uvicorn
from router import router
import nest_asyncio

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

app = FastAPI(title="LLM Doc QA Backend", version="1.0")
app.include_router(router, prefix="/api/v1")

@app.get("/")  # root route
def read_root():
    return {"message": "Welcome to my FastAPI app!"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)