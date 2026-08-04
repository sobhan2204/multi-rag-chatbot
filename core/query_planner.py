import json
import logging
import re
from typing import Dict, Any

from config.pipeline_config import PipelineConfig

logger = logging.getLogger(__name__)

class QueryPlanner:
    """
    Classifies queries to determine the execution strategy.
    Categories: FACT_LOOKUP, AGGREGATION, ENUMERATION, MULTI_HOP_REASONING, etc.
    """
    def __init__(self, config: PipelineConfig, llm_client):
        self.config = config
        self.llm_client = llm_client

    def extract_json(self, text: str) -> dict:
        text = text.strip()
        if not text:
            raise ValueError("Empty response received from LLM")
            
        # Strip markdown json blocks
        text = re.sub(r'^```json\s*', '', text)
        text = re.sub(r'^```\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
        
        # Sometimes there's conversational text before the json block
        start_idx = text.find('{')
        end_idx = text.rfind('}')
        if start_idx != -1 and end_idx != -1 and end_idx >= start_idx:
            text = text[start_idx:end_idx+1]
            
        return json.loads(text)

    def plan_query(self, query: str) -> Dict[str, Any]:
        """Classify the query using the configured LLM."""
        if not self.config.planner.enabled or not self.llm_client:
             return self._fallback_plan(query)

        system_prompt = """
        You are a Query Planner for a Multi-RAG system.
        Classify the user query into exactly ONE of the following categories:
        - FACT_LOOKUP: Standard information retrieval (e.g., "What is the deductible?")
        - AGGREGATION: Counting or summarizing quantities across documents (e.g., "How many types of hospitals?")
        - ENUMERATION: Listing entities (e.g., "List all diseases mentioned.")
        - MULTI_HOP_REASONING: Requires combining information from different places (e.g., "Does the policy that covers X also cover Y?")
        - DEFINITION: Asking for the meaning of a term.

        Return ONLY a JSON object with:
        {
          "category": "ONE_OF_THE_ABOVE",
          "entities": ["entity1", "entity2"] (list of key subjects),
          "requires_agentic_loop": boolean (true if multi-hop),
          "top_k": integer (recommended retrieval depth)
        }
        """

        try:
             logger.debug(f"Planner model: {self.config.planner.model}")
             response = self.llm_client.chat.completions.create(
                 model=self.config.planner.model,
                 messages=[
                     {"role": "system", "content": system_prompt},
                     {"role": "user", "content": f"Query: {query}"}
                 ],
                 temperature=0,
                 response_format={"type": "json_object"}
             )
             raw_content = response.choices[0].message.content
             plan = self.extract_json(raw_content)
             logger.info(f"Query Plan generated: {plan}")
             return plan
        except Exception as e:
             logger.warning(f"Query planner LLM failed: {e}. Falling back to rule-based.")
             return self._fallback_plan(query)

    def _fallback_plan(self, query: str) -> Dict[str, Any]:
        q = query.lower()
        plan = {
            "entities": [],
            "requires_agentic_loop": False,
            "top_k": self.config.retrieval.top_k
        }
        
        if "how many" in q or "count" in q:
            plan["category"] = "AGGREGATION"
        elif any(phrase in q for phrase in ["list all", "show all", "what are all", "different kinds", "types of"]):
            plan["category"] = "ENUMERATION"
        elif "what is" in q or "define" in q:
             plan["category"] = "DEFINITION"
        else:
             plan["category"] = "FACT_LOOKUP"
             
        logger.info(f"Fallback Query Plan generated: {plan}")
        return plan
