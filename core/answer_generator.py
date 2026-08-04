import logging
from typing import List, Dict, Any, Tuple

from config.pipeline_config import PipelineConfig

logger = logging.getLogger(__name__)

class AnswerGenerator:
    """
    Generates answers forcing citations and calculates explainable confidence scores.
    """
    def __init__(self, config: PipelineConfig, llm_client):
         self.config = config
         self.llm_client = llm_client

    def generate(self, query: str, context: List[str], plan: Dict[str, Any] = None) -> Tuple[str, float, bool]:
         """
         Generate an answer strictly based on context, ensuring citations.
         Returns (answer_text, confidence_score, is_error). ``is_error`` is True only
         when the LLM call itself failed (e.g. network/rate-limit error) - callers
         should treat that distinctly from a legitimate (if low-confidence) answer.
         """
         if not context:
              return "I could not find relevant evidence in the documents to answer your question.", 0.0, False

         context_str = "\n".join(context)
         
         # Truncate context if needed
         max_chars = 30000 
         if len(context_str) > max_chars:
              context_str = context_str[:max_chars] + "\n...[Context truncated]"

         prompt = f"""
         You are a precise, evidence-based AI assistant and an expert domain advisor.
         Answer the user's question comprehensively based strictly on the provided Context.
         
         CRITICAL RULES:
         1. Provide a detailed, explanatory answer. Do not just copy-paste the context verbatim. Synthesize the information, explain the concepts clearly, and elaborate on the details found in the documents to give the user a complete understanding.
         2. Do NOT hallucinate. If the context does not contain the answer, say "I cannot answer based on the provided documents."
         3. You MUST cite your sources for every factual claim.
         4. Format citations as [Source: FileName].
         
         Context:
         {context_str}
         
         Question: {query}
         """

         try:
              response = self.llm_client.chat.completions.create(
                   model=self.config.groq_model,
                   messages=[{"role": "user", "content": prompt}],
                   temperature=0.1,
                   max_tokens=2000
              )
              answer = response.choices[0].message.content.strip()
              confidence = self._calculate_confidence(answer, context)
              return answer, confidence, False
         except Exception as e:
              logger.error(f"Answer generation failed: {e}")
              return f"Error generating answer: {e}", 0.0, True

    def _calculate_confidence(self, answer: str, context: List[str]) -> float:
         """
         Calculate confidence based on citation presence and denial phrases.
         """
         answer_lower = answer.lower()
         
         # Severe penalty for "cannot answer"
         denial_phrases = ["cannot answer based on", "not found", "no information"]
         if any(phrase in answer_lower for phrase in denial_phrases):
              return 0.1
              
         # Check for citations
         has_citation = "[source:" in answer_lower
         
         # Base score
         score = 0.5
         
         # Boost if citations are present
         if has_citation:
              score += 0.3
              
         # Boost if context size was decent
         if len(context) > 2:
              score += 0.1
              
         return min(score, 1.0)
