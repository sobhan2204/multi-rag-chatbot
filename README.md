# 🤖 Multi-RAG Chatbot

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![RAG](https://img.shields.io/badge/RAG-Enabled-orange.svg)
![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)

**A comprehensive framework for exploring Retrieval-Augmented Generation (RAG) with multiple retrieval algorithms**

[🚀 Quick Start](#-quick-start) • [📖 Documentation](#-features) • [🛠️ Installation](#️-installation) • [💡 Examples](#-usage-examples) • [🤝 Contributing](#-contributing)

</div>

---

## 🎯 Overview

Multi-RAG Chatbot is a powerful framework designed to compare and evaluate different Retrieval-Augmented Generation approaches on document-based question-answering tasks. This project enables researchers and developers to experiment with various RAG methodologies including **Probabilistic RAG**, **Graph RAG**, and **BM25** retrieval algorithms.

### ✨ Key Highlights

- 🔍 **Multiple Retrieval Algorithms** - Compare Probabilistic RAG, Graph RAG, and BM25
- 📊 **Performance Evaluation** - Built-in metrics and comparison tools
- 🎛️ **Flexible Configuration** - Easy-to-customize parameters and models
- 📚 **Document Processing** - Support for multiple document formats
- 🌐 **Interactive Interface** - User-friendly chat interface
- ⚡ **Optimized Performance** - Efficient retrieval and generation pipeline

---

## 🌟 Features

### 🔧 RAG Algorithms
| Algorithm | Description | Best For |
|-----------|-------------|----------|
| **Probabilistic RAG** | Uses embedding similarity and probability scoring | General-purpose QA |
| **Graph RAG** | Leverages knowledge graphs for contextual retrieval | Complex, interconnected documents |
| **BM25** | Traditional keyword-based retrieval with TF-IDF | Keyword-specific queries |

### 📋 Core Capabilities
- ✅ Multi-format document ingestion (PDF, TXT, DOCX, MD)
- ✅ Real-time performance comparison
- ✅ Customizable embedding models
- ✅ Advanced chunking strategies
- ✅ Interactive evaluation dashboard
- ✅ Export results and metrics

---

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git

### Quick Installation

```bash
# Clone the repository
git clone https://github.com/sobhan2204/multi-rag-chatbot.git
cd multi-rag-chatbot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```


## 🚀 Quick Start

### 1. Basic Setup
```python
from multi_rag_chatbot import MultiRAGChatbot

# Initialize the chatbot
chatbot = MultiRAGChatbot(
    algorithms=['probabilistic', 'graph', 'bm25']
)

# Load documents
chatbot.load_documents('path/to/your/documents/')
```

### 2. Run Comparisons
```python
# Ask a question and compare results
question = "What are the main benefits of renewable energy?"
results = chatbot.compare_algorithms(question)

# View performance metrics
chatbot.display_metrics()
```

## 💡 Usage Examples

### Example 1: Document Analysis
```python
from multi_rag_chatbot import DocumentProcessor, RAGComparator

# Process documents
processor = DocumentProcessor()
documents = processor.load_from_directory("./documents/")

# Compare algorithms
comparator = RAGComparator(documents)
results = comparator.evaluate_question("Explain the concept of machine learning")

print(f"Best performing algorithm: {results.best_algorithm}")
print(f"Confidence score: {results.confidence}")
```

### Example 2: Custom Configuration
```python
config = {
    'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
    'chunk_size': 1000,
    'overlap': 200,
    'top_k': 5
}

chatbot = MultiRAGChatbot(config=config)
chatbot.add_algorithm('custom_rag', CustomRAGImplementation())
```

---

## 📊 Performance Metrics

The framework provides comprehensive evaluation metrics:

- **Retrieval Accuracy**: Measures relevance of retrieved documents
- **Response Quality**: BLEU, ROUGE scores for generated answers  
- **Latency**: Response time for each algorithm
- **Resource Usage**: Memory and CPU consumption
- **Coherence Score**: Semantic consistency of responses

---

## 📁 Project Structure

```
multi-rag-chatbot/
├── 📂 src/
│   ├── 🧠 algorithms/
│   │   ├── probabilistic_rag.py
│   │   ├── graph_rag.py
│   │   └── bm25_rag.py
│   ├── 📊 evaluation/
│   │   ├── metrics.py
│   │   └── comparator.py
│   ├── 📄 document_processing/
│   │   ├── loader.py
│   │   └── chunker.py
│   └── 🌐 interface/
│       ├── streamlit_app.py
│       └── api.py
├── 📖 docs/
│   ├── api_reference.md
│   └── tutorials/
├── 🧪 tests/
├── 📊 examples/
├── 📋 requirements.txt
└── 📖 README.md
```

---

## ⚙️ Configuration

### Environment Variables
Create a `.env` file in the root directory:

```env
# OpenAI API (optional)
OPENAI_API_KEY=your_api_key_here

# Hugging Face Token (optional)
HUGGINGFACE_TOKEN=your_token_here

# Database Configuration
VECTOR_DB_PATH=./data/vectordb
GRAPH_DB_URL=bolt://localhost:7687

# Performance Settings
MAX_CONCURRENT_REQUESTS=10
CACHE_SIZE=1000
```

### Model Configuration
```yaml
# config.yaml
models:
  embedding: "sentence-transformers/all-MiniLM-L6-v2"
  llm: "gpt-3.5-turbo"
  
retrieval:
  chunk_size: 1000
  chunk_overlap: 200
  top_k: 5
  algorithms:
    - probabilistic
    - graph
    - bm25
```

---

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/

# Run specific test categories
python -m pytest tests/test_algorithms.py -v
python -m pytest tests/test_evaluation.py -v

# Run with coverage
python -m pytest --cov=src tests/
```

---

## 🤝 Contributing

We welcome contributions! Here's how you can help:

### 🐛 Bug Reports
- Use the [issue tracker](https://github.com/sobhan2204/multi-rag-chatbot/issues)
- Include detailed reproduction steps
- Provide system information

### ✨ Feature Requests
- Check existing issues first
- Describe the use case clearly
- Consider submitting a pull request

### 🔧 Development Setup
```bash
# Fork and clone the repository
git clone https://github.com/your-username/multi-rag-chatbot.git

# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests before submitting
python -m pytest
```

### 📋 Contribution Guidelines
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [LangChain](https://langchain.com/) for RAG framework inspiration
- [Hugging Face](https://huggingface.co/) for transformer models
- The open-source community for various libraries and tools

---

## 📞 Support & Contact

- 📧 **Email**: sobhan2204@example.com
- 🐛 **Issues**: [GitHub Issues](https://github.com/sobhan2204/multi-rag-chatbot/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/sobhan2204/multi-rag-chatbot/discussions)

---

<div align="center">

**⭐ If this project helped you, please consider giving it a star! ⭐**

[⬆ Back to Top](#-multi-rag-chatbot)

</div>