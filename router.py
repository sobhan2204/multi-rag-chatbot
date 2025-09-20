
import os
import sys
import types
import importlib.machinery
import tempfile
import requests
from pathlib import Path
    
    # disable TensorFlow completely
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF_IMPORT"] = "1"
    
    # create a fake tensorflow module with a proper spec
fake_tf = types.ModuleType("tensorflow")
fake_tf.__spec__ = importlib.machinery.ModuleSpec("tensorflow", None)
sys.modules["tensorflow"] = fake_tf
    
from fastapi import APIRouter, HTTPException, Request, Depends, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List,Union
from query_final import query_pipeline
from utils import download_file_and_chunk
from dotenv import load_dotenv
    
router = APIRouter()
bearer = HTTPBearer()
    
load_dotenv()
    
VALID_TOKEN = os.getenv("VALID_TOKEN")


class QARequest(BaseModel):
        documents: Union[str, None] # URL of the document or just raw text 
        questions: List[str]
    
    
class QAResponse(BaseModel):
        answers: List[str]
    
    
def validate_token(credentials: HTTPAuthorizationCredentials = Depends(bearer)):
        if not VALID_TOKEN:
            raise HTTPException(status_code=500, detail="Server configuration error: Missing VALID_TOKEN")
        if credentials.credentials != VALID_TOKEN:
            raise HTTPException(status_code=403, detail="Invalid or missing token")
        return credentials.credentials
    
    
def get_file_extension_from_url(url: str) -> str:
        """Get file extension from URL or content type"""
        url = url.lower()
        if url.endswith('.pdf'):
            return '.pdf'
        elif url.endswith('.docx'):
            return '.docx'
        elif url.endswith('.eml'):
            return '.eml'
        else:
            try:
                resp = requests.head(url, timeout=3)
                content_type = resp.headers.get('content-type', '').lower()
                if 'pdf' in content_type:
                    return '.pdf'
                elif 'msword' in content_type or 'officedocument' in content_type:
                    return '.docx'
                elif 'message/rfc822' in content_type or 'eml' in content_type:
                    return '.eml'
            except Exception:
                pass
            return '.pdf'  # default to prevent breaking
    
    
def process_document_and_qa(document_url: str, questions: List[str]) -> List[str]:
        """Process document and answer questions"""
        temp_file_path = None
        try:
            if document_url.startswith("http://") or document_url.startswith("https://"):
                # Create data directory if it doesn't exist
                data_dir = Path("data")
                data_dir.mkdir(exist_ok=True)
                
                # Get file extension and create temp file
                ext = get_file_extension_from_url(document_url)
                temp_file = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
                temp_file_path = temp_file.name
                
                # Download the document
                response = requests.get(document_url, timeout=30)
                response.raise_for_status()
                
                temp_file.write(response.content)
                temp_file.close()
                
                # Process the document (use temp file path instead of hardcoded path)
                download_file_and_chunk(temp_file_path)
                
                # Answer questions
                answers = []
                for question in questions:
                    answer = query_pipeline(question)
                    answers.append(answer)
                
                return answers
            else:
                # Assume raw text input
                with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w", encoding="utf-8") as temp_file:
                    temp_file.write(document_url)
                    temp_file_path = temp_file.name
                download_file_and_chunk(temp_file_path)
                # Answer questions
                answers = []
                for question in questions:
                    answer = query_pipeline(question)
                    answers.append(answer)
                return answers
    
        except requests.RequestException as e:
            raise HTTPException(status_code=400, detail=f"Failed to download document: {str(e)}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")
        finally:
            # Cleanup temp file
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except Exception:
                    pass  # Ignore cleanup errors
    
    
@router.post("/medical_rag/run", response_model=QAResponse)
def run_qa_post(payload: QARequest, token: str = Depends(validate_token)):
        """Run Q&A on documents via POST request"""
        answers = process_document_and_qa(payload.documents, payload.questions)
        return {"answers": answers}
    
    
@router.get("/medical_rag/run", response_model=QAResponse)
def run_qa_get(
        documents: str = Query(..., description="URL of the document to process"),
        questions: List[str] = Query(..., description="List of questions to ask"),
        token: str = Depends(validate_token)):
        """Run Q&A on documents via GET request"""
        answers = process_document_and_qa(documents, questions)
        return {"answers": answers}