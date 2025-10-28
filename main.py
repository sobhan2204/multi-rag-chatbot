from query_with_BM25 import HybridBM25RAG
from query_final_KG import EnhancedKnowledgeGraph
from rich.console import Console
console = Console()

rag = {HybridBM25RAG(
    data_folder="data",
    chunk_size=800,
    chunk_overlap=100,
    min_score_threshold=0.1
),
         EnhancedKnowledgeGraph(
             data_folder="data",
             
}

if __name__ == "__main__":
    console.print("[bold magenta]Welcome to the RAG-based Insurance Query System !  [/]")
    console.print("[green]Type 'q', 'quit', or 'exit' to leave the program[/]!")
    while True:
     test_query = input("Enter your query: ")
     if(test_query.lower() in ['q' , 'quit' , 'exit']):
        console.print("[red]THANK YOU ! FOR USING OUR SERVICE[/]")
        break
     answer = rag.query(test_query, top_k=3, debug=False)
     print(f"Query: {test_query}")
     print(f"Answer: {answer}")
