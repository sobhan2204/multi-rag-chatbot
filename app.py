"""
Flask Backend Server for Multi-RAG Chatbot
Integrates with your existing RAG pipeline and PDF scraper
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import sys
from datetime import datetime
import traceback

from web_scraper import PDFScraper
from main import IntegratedRAGPipeline

app = Flask(__name__, static_folder='build', static_url_path='')
CORS(app)

# Initialize RAG Pipeline (singleton)
rag_pipeline = None
DATA_FOLDER = 'data'

def get_rag_pipeline():
    """Get or create RAG pipeline instance"""
    global rag_pipeline
    if rag_pipeline is None:
        rag_pipeline = IntegratedRAGPipeline(data_folder=DATA_FOLDER)
    return rag_pipeline

# Ensure data folder exists
os.makedirs(DATA_FOLDER, exist_ok=True)

@app.route('/')
def index():
    """Serve React app"""
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/scrape-pdfs', methods=['POST'])
def scrape_pdfs():
    """
    Endpoint to scrape PDFs from a given URL
    Request body: {"url": "https://example.com"}
    """
    try:
        data = request.get_json()
        url = data.get('url', '').strip()
        
        if not url:
            return jsonify({
                'success': False,
                'message': 'URL is required'
            }), 400
        
        # Add https:// if not present
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        # Initialize scraper
        scraper = PDFScraper(url, data_folder=DATA_FOLDER)
        
        # Scrape PDFs
        downloaded_files = scraper.scrape_all_pdfs(delay=1)
        
        if downloaded_files:
            # Reset RAG pipeline to reload with new files
            global rag_pipeline
            rag_pipeline = None
            
            return jsonify({
                'success': True,
                'count': len(downloaded_files),
                'files': [os.path.basename(f) for f in downloaded_files],
                'message': f'Successfully downloaded {len(downloaded_files)} PDFs'
            })
        else:
            return jsonify({
                'success': False,
                'message': 'No PDFs found or download failed'
            }), 400
    
    except Exception as e:
        print(f"Error in scrape_pdfs: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        }), 500

@app.route('/api/query-rag', methods=['POST'])
def query_rag():
    """
    Endpoint to query the RAG pipeline
    Request body: {"query": "user question", "show_comparison": false}
    """
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        show_comparison = data.get('show_comparison', False)
        
        if not query:
            return jsonify({
                'success': False,
                'message': 'Query is required'
            }), 400
        
        # Get RAG pipeline
        pipeline = get_rag_pipeline()
        
        # Query all models
        responses = pipeline.query_all_models(query)
        
        if not responses:
            return jsonify({
                'success': False,
                'message': 'No RAG models available or query failed'
            }), 500
        
        # Sort by confidence score
        responses.sort(key=lambda x: x.confidence_score, reverse=True)
        best_response = responses[0]
        
        # Prepare comparison data
        comparison = None
        if show_comparison and len(responses) > 1:
            comparison = [
                {
                    'model': resp.model_name,
                    'confidence_score': round(resp.confidence_score, 3),
                    'retrieval_quality': round(resp.retrieval_quality, 3),
                    'answer_quality': round(resp.answer_quality, 3),
                    'is_winner': i == 0
                }
                for i, resp in enumerate(responses)
            ]
        
        return jsonify({
            'success': True,
            'answer': best_response.answer,
            'model_used': best_response.model_name,
            'confidence_score': round(best_response.confidence_score, 3),
            'comparison': comparison
        })
    
    except Exception as e:
        print(f"Error in query_rag: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        }), 500

@app.route('/api/list-files', methods=['GET'])
def list_files():
    """
    Endpoint to list all files in the data folder
    """
    try:
        files = []
        
        if os.path.exists(DATA_FOLDER):
            for filename in os.listdir(DATA_FOLDER):
                filepath = os.path.join(DATA_FOLDER, filename)
                
                if os.path.isfile(filepath):
                    # Get file stats
                    stat = os.stat(filepath)
                    size_mb = stat.st_size / (1024 * 1024)
                    modified_date = datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d')
                    
                    files.append({
                        'name': filename,
                        'size': f'{size_mb:.2f} MB',
                        'date': modified_date
                    })
        
        # Sort by date (newest first)
        files.sort(key=lambda x: x['date'], reverse=True)
        
        return jsonify({
            'success': True,
            'files': files,
            'count': len(files)
        })
    
    except Exception as e:
        print(f"Error in list_files: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        pipeline = get_rag_pipeline()
        return jsonify({
            'status': 'healthy',
            'models_available': pipeline.models,
            'data_folder': DATA_FOLDER
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

# Error handlers
@app.errorhandler(404)
def not_found(e):
    return send_from_directory(app.static_folder, 'index.html')

@app.errorhandler(500)
def internal_error(e):
    return jsonify({
        'success': False,
        'message': 'Internal server error'
    }), 500

if __name__ == '__main__':
    print("=" * 60)
    print("Multi-RAG Chatbot Server")
    print("=" * 60)
    print(f"Data folder: {DATA_FOLDER}")
    print("Initializing RAG pipeline...")
    
    # Initialize pipeline on startup
    try:
        pipeline = get_rag_pipeline()
        print(f"✓ RAG pipeline initialized with models: {', '.join(pipeline.models)}")
    except Exception as e:
        print(f"⚠ Warning: Could not initialize RAG pipeline: {e}")
        print("  The server will still start, but RAG queries may fail.")
    
    print("\nStarting server...")
    print("Server running at: http://localhost:5000")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)