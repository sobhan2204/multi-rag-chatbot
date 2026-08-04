import json
import logging

# Configure logging
logger = logging.getLogger(__name__)

def classify_query(query: str, groq_client) -> dict:
    """
    Classifies the user query intent and extracts retrieval parameters using Groq.
    """
    if not groq_client:
        # Fallback if no LLM client is available
        return {
            "intent": "general",
            "semantic_subject": query,
            "exhaustive": False,
            "retrieval_top_k": 5,
            "group_by_source": False,
            "llm_instruction": "Answer the question clearly based on the context."
        }

    system_prompt = """
Return ONLY a valid JSON object with exactly these fields:
{
  "intent": string,
  "semantic_subject": string,
  "exhaustive": boolean,
  "retrieval_top_k": integer,
  "group_by_source": boolean,
  "llm_instruction": string
}

Field definitions:
- intent: a single word describing what the user fundamentally wants — infer this freely from the query.
- semantic_subject: the normalized, cleaned subject of the query stripped of filler words, pronouns, and intent signals.
- exhaustive: true if the user's query implies they want complete/all results, false if they want the single best answer.
- retrieval_top_k: an integer recommended based on how broad the search needs to be (min 5, max 40). Scale with exhaustive and query breadth.
- group_by_source: true if results from different source documents should be presented separately in the final answer.
- llm_instruction: a single precise sentence telling the answer-generation LLM exactly how to structure and present its response for this specific query.
"""

    user_prompt = f"Query: {query}"

    def get_json_response(prompt):
        resp = groq_client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )
        return resp.choices[0].message.content.strip()

    try:
        content = get_json_response(user_prompt)
        intent_dict = json.loads(content)
    except Exception as e:
        logger.warning(f"Failed to parse classification JSON, retrying once. Error: {e}")
        try:
            content = get_json_response(f"Return ONLY valid JSON for this query: {query}")
            intent_dict = json.loads(content)
        except Exception as retry_e:
            logger.error(f"Retry failed to parse classification JSON. Error: {retry_e}")
            raise retry_e

    logger.debug(f"Query Intelligence Output: {intent_dict}")
    return intent_dict
