import logging
from typing import Dict, Any, List

from config.pipeline_config import PipelineConfig

logger = logging.getLogger(__name__)

class AgenticRetriever:
    """
    Manages multi-step retrieval loops for complex queries.
    """
    def __init__(self, config: PipelineConfig, llm_client, hybrid_retriever):
        self.config = config
        self.llm_client = llm_client
        self.retriever = hybrid_retriever

    def execute_loop(self, query: str, plan: Dict[str, Any]) -> List[str]:
        """
        Executes a retrieval loop, querying, assessing gaps, and querying again.
        """
        max_steps = self.config.agentic.max_steps if self.config.agentic.enabled else 1
        
        logger.info(f"Starting agentic retrieval loop (max_steps={max_steps}) for query: {query}")
        
        current_query = query
        accumulated_context = []
        
        for step in range(max_steps):
            logger.info(f"--- Agentic Loop Step {step+1} ---")
            
            # 1. Retrieve
            context = self.retriever.retrieve_and_fuse(current_query, plan.get("entities"))
            accumulated_context.extend(context)
            
            # Deduplicate accumulated context
            accumulated_context = list(dict.fromkeys(accumulated_context))
            
            # 2. Assess completion (Stop early if not agentic or if we hit the limit)
            if not self.config.agentic.enabled or not plan.get("requires_agentic_loop") or step == max_steps - 1:
                 break
                 
            # 3. LLM Gap Analysis (Decide if we need more info and formulate next query)
            next_query = self._assess_gaps_and_replan(query, accumulated_context)
            if not next_query:
                 logger.info("Agent determined sufficient context has been gathered.")
                 break
                 
            logger.info(f"Agent formulated follow-up query: {next_query}")
            current_query = next_query
            
        return accumulated_context

    def _assess_gaps_and_replan(self, original_query: str, current_context: List[str]) -> str:
        """
        Asks the LLM if the current context is sufficient. If not, returns a new query.
        Returns None if sufficient.
        """
        if not self.llm_client:
             return None
             
        context_str = "\n".join(current_context[:10]) # Limit to top 10 for assessment to save tokens
        
        system_prompt = """
        You are a Retrieval Assessment Agent.
        Analyze the original question and the current context.
        Determine if the context contains enough information to fully answer the question.
        
        If YES, return "SUFFICIENT".
        If NO, return a SINGLE SPECIFIC follow-up query to find the missing information.
        Do not return JSON, just the string "SUFFICIENT" or the new query string.
        """
        
        prompt = f"""
        Original Question: {original_query}
        
        Current Context:
        {context_str}
        
        Is the context sufficient?
        """
        
        try:
             response = self.llm_client.chat.completions.create(
                 model="llama-3.3-70b-versatile",
                 messages=[
                     {"role": "system", "content": system_prompt},
                     {"role": "user", "content": prompt}
                 ],
                 temperature=0
             )
             result = response.choices[0].message.content.strip()
             if result.upper() == "SUFFICIENT" or len(result) < 5:
                  return None
             return result
        except Exception as e:
             logger.error(f"Gap assessment failed: {e}")
             return None
