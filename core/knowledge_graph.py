import logging
import sqlite3
from typing import List, Dict, Any, Tuple

from core.ingestion import CorpusAnalytics

logger = logging.getLogger(__name__)

class KnowledgeGraph:
    """
    Dynamic Knowledge Graph built over the Corpus Analytics index.
    Provides graph traversal and neighborhood search capabilities.
    """
    def __init__(self, analytics: CorpusAnalytics):
        self.analytics = analytics
        
    def ensure_loaded(self):
         cursor = self.analytics.conn.cursor()
         cursor.execute("SELECT COUNT(*) FROM relationships")
         if cursor.fetchone()[0] == 0:
              self.analytics.load_from_disk()

    def get_neighborhood(self, entity_text: str, depth: int = 1) -> List[Dict[str, Any]]:
        """
        Retrieves the 1-hop or N-hop neighborhood for a given entity.
        Returns a list of relationships.
        """
        self.ensure_loaded()
        cursor = self.analytics.conn.cursor()
        
        # Simple 1-hop search for now
        cursor.execute('''
            SELECT source_entity, relation, target_entity, source_document
            FROM relationships
            WHERE source_entity = ? COLLATE NOCASE OR target_entity = ? COLLATE NOCASE
        ''', (entity_text.lower(), entity_text.lower()))
        
        results = cursor.fetchall()
        neighborhood = []
        for row in results:
             neighborhood.append({
                 "source": row[0],
                 "relation": row[1],
                 "target": row[2],
                 "document": row[3]
             })
        return neighborhood
        
    def query_graph(self, query: str, extracted_entities: List[str]) -> List[str]:
        """
        Takes a natural language query and key entities, and returns contextual strings 
        derived from graph relationships to be fed into the LLM context.
        """
        self.ensure_loaded()
        context_strings = []
        for entity in extracted_entities:
            neighborhood = self.get_neighborhood(entity)
            for rel in neighborhood:
                 context_strings.append(
                     f"Graph Fact: {rel['source'].title()} {rel['relation'].lower()} {rel['target'].title()} (Source: {rel['document']})"
                 )
                 
        return list(set(context_strings)) # Deduplicate
