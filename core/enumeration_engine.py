import logging
import json
import re
from typing import Dict, Any, List, Tuple

from config.pipeline_config import PipelineConfig

logger = logging.getLogger(__name__)

class EnumerationEngine:
    """
    Dedicated engine for ENUMERATION queries.
    Instead of relying on rigid SpaCy NER entities, it retrieves relevant chunks 
    using the HybridRetriever and then uses the LLM to extract and deduplicate
    the requested semantic entities from the context.
    """
    def __init__(self, config: PipelineConfig, llm_client, hybrid_retriever):
        self.config = config
        self.llm_client = llm_client
        self.retriever = hybrid_retriever

    def _extract_json(self, text: str) -> dict:
        text = text.strip()
        if not text:
            raise ValueError("Empty response received from LLM")
            
        # Strip markdown json blocks
        text = re.sub(r'^```json\s*', '', text)
        text = re.sub(r'^```\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
        
        # Ensure we only parse the JSON object
        start_idx = text.find('{')
        end_idx = text.rfind('}')
        if start_idx != -1 and end_idx != -1 and end_idx >= start_idx:
            text = text[start_idx:end_idx+1]
            
        return json.loads(text)

    def execute_enumeration(self, query: str, plan: Dict[str, Any]) -> Tuple[str, float, Dict[str, Any]]:
        """
        Executes an enumeration query by retrieving context and using the LLM to list items.
        Returns (formatted_answer_string, confidence_score, structured_json_data)
        """
        logger.info(f"Executing Enumeration Engine for query: '{query}'")
        
        # 1. Retrieve chunks
        # Force top_k from plan if available, else use config
        top_k = plan.get("top_k", self.config.retrieval.top_k)
        
        # Temporarily override config top_k for this retrieval
        original_top_k = self.config.retrieval.top_k
        self.config.retrieval.top_k = top_k
        try:
             context_chunks = self.retriever.retrieve_and_fuse(query, plan.get("entities"))
        finally:
             self.config.retrieval.top_k = original_top_k
             
        logger.info(f"Enumeration Engine retrieved {len(context_chunks)} chunks.")

        if not context_chunks:
             return "I could not find relevant evidence in the documents to extract the requested list.", 0.0, {}

        # 2. Extract and format with LLM
        context_str = "\n---\n".join(context_chunks)
        max_chars = 30000 
        if len(context_str) > max_chars:
             context_str = context_str[:max_chars] + "\n...[Context truncated]"

        system_prompt = """
        You are an expert Data Extraction AI. 
        Given the retrieved document excerpts, identify and list all unique entities that answer the user's request. 
        Remove duplicates. 
        Only include entities explicitly mentioned in the provided context. 
        Do not invent information.
        
        You must return the result as a valid JSON object matching this schema exactly:
        {
          "answer": "A brief summary statement (e.g. 'I found 5 types of hospitals mentioned in the documents.')",
          "items": [
              {
                 "name": "The extracted entity name",
                 "source": "The source document name found in the context (if available, else 'Unknown')"
              }
          ]
        }
        """

        user_prompt = f"User Query: {query}\n\nContext:\n{context_str}"

        try:
             response = self.llm_client.chat.completions.create(
                 model=self.config.groq_model,
                 messages=[
                     {"role": "system", "content": system_prompt},
                     {"role": "user", "content": user_prompt}
                 ],
                 temperature=0.1,
                 response_format={"type": "json_object"}
             )
             
             raw_content = response.choices[0].message.content
             result_json = self._extract_json(raw_content)
             
             items = result_json.get("items", [])
             logger.info(f"Enumeration Engine extracted {len(items)} entities.")
             
             # Format for console
             answer_text = result_json.get("answer", "Here are the extracted items:")
             formatted_lines = [answer_text, ""]
             for idx, item in enumerate(items, 1):
                  name = item.get("name", "Unknown")
                  source = item.get("source", "Unknown")
                  formatted_lines.append(f"{idx}. **{name}** (Source: {source})")
                  
             final_answer = "\n".join(formatted_lines)
             
             # Confidence heuristic: if items were found, we are confident.
             confidence = 0.8 if len(items) > 0 else 0.2
             
             return final_answer, confidence, result_json

        except Exception as e:
             logger.error(f"Enumeration generation failed: {e}")
             return f"Failed to extract list: {e}", 0.0, {}
