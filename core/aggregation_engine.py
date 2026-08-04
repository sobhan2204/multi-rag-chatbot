import logging
from typing import Dict, Any, List

from core.ingestion import CorpusAnalytics

logger = logging.getLogger(__name__)

class AggregationEngine:
    """
    Handles queries that require counting or enumerating entities across the corpus.
    Relies on the structured SQLite index in CorpusAnalytics.
    """
    def __init__(self, analytics: CorpusAnalytics):
        self.analytics = analytics
        
    def map_query_to_entity_type(self, query: str) -> str:
        """
        Maps a natural language query to an expected entity type.
        This is a simplistic rule-based mapping, but could be enhanced with an LLM.
        """
        query_lower = query.lower()
        if "organization" in query_lower or "hospital" in query_lower or "company" in query_lower:
            return "ORG"
        if "person" in query_lower or "people" in query_lower or "name" in query_lower:
            return "PERSON"
        if "location" in query_lower or "place" in query_lower or "city" in query_lower or "country" in query_lower:
            return "GPE"
        if "disease" in query_lower or "condition" in query_lower or "illness" in query_lower:
            # Spacy might classify some diseases as ORG or generic. We might need custom extractors for domain specific types.
            # Returning a wildcard or generic type if unknown.
            return "DISEASE" # Assuming a custom extractor might add this
            
        # Fallback to look at all generic entities or ask LLM
        return None

    def execute_aggregation(self, query: str, entity_type: str = None) -> str:
        """
        Executes an aggregation query. If entity_type is not provided, it tries to infer it.
        Returns a formatted string containing the answer.
        """
        if not entity_type:
            entity_type = self.map_query_to_entity_type(query)
            
        if not entity_type:
            return "I could not determine the specific type of entity to aggregate based on your query."
            
        logger.info(f"Aggregating entities of type: {entity_type}")
        
        # If the index is empty, try to load it
        cursor = self.analytics.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM entities")
        count = cursor.fetchone()[0]
        if count == 0:
             self.analytics.load_from_disk()
             
        results = self.analytics.query_aggregate(entity_type)
        
        total_unique = results.get("total_unique", 0)
        items = results.get("items", [])
        
        if total_unique == 0:
            return f"I found no entities of type '{entity_type}' in the indexed documents."
            
        # Format the response
        response_parts = [f"Found {total_unique} unique '{entity_type}' entities across the documents."]
        response_parts.append("\nTop results:")
        
        # Show top 20 or fewer
        for item in items[:20]:
            sources_str = ", ".join(set(item['sources']))
            response_parts.append(f"- **{item['entity'].title()}** (Mentions: {item['count']}, Sources: {sources_str})")
            
        if total_unique > 20:
             response_parts.append(f"\n... and {total_unique - 20} more.")
             
        return "\n".join(response_parts)
