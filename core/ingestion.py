import os
import json
import sqlite3
import re
from typing import List, Dict, Any, Tuple
import logging

import spacy

logger = logging.getLogger(__name__)

class CorpusAnalytics:
    """
    Structured index for corpus analytics (Aggregation and Enumeration).
    Uses an in-memory SQLite database for deterministic querying.
    """
    def __init__(self, db_path: str = ":memory:"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_db()
        # Fallback to local file if needed
        self.persist_path = "data/corpus_analytics.json"

    def _init_db(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS entities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_text TEXT,
                entity_type TEXT,
                source_document TEXT,
                chunk_id INTEGER,
                page_number INTEGER,
                section TEXT
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS relationships (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_entity TEXT,
                relation TEXT,
                target_entity TEXT,
                source_document TEXT
            )
        ''')
        self.conn.commit()

    def add_entity(self, entity_text: str, entity_type: str, source_doc: str, chunk_id: int, page_num: int = None, section: str = None):
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO entities (entity_text, entity_type, source_document, chunk_id, page_number, section)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (entity_text.lower(), entity_type, source_doc, chunk_id, page_num, section))
        self.conn.commit()

    def add_relationship(self, source_entity: str, relation: str, target_entity: str, source_doc: str):
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO relationships (source_entity, relation, target_entity, source_document)
            VALUES (?, ?, ?, ?)
        ''', (source_entity.lower(), relation, target_entity.lower(), source_doc))
        self.conn.commit()

    def query_aggregate(self, entity_type: str) -> Dict[str, Any]:
        """Aggregate occurrences of an entity type across the corpus."""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT entity_text, COUNT(*) as count, GROUP_CONCAT(DISTINCT source_document) 
            FROM entities 
            WHERE entity_type = ? COLLATE NOCASE
            GROUP BY entity_text
            ORDER BY count DESC
        ''', (entity_type,))
        results = cursor.fetchall()
        
        aggregated = []
        for row in results:
            aggregated.append({
                "entity": row[0],
                "count": row[1],
                "sources": row[2].split(',') if row[2] else []
            })
        return {"entity_type": entity_type, "total_unique": len(aggregated), "items": aggregated}

    def save_to_disk(self):
        """Save a snapshot to disk for persistence across runs."""
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM entities')
        entities = cursor.fetchall()
        cursor.execute('SELECT * FROM relationships')
        relations = cursor.fetchall()
        
        data = {"entities": entities, "relationships": relations}
        os.makedirs(os.path.dirname(self.persist_path), exist_ok=True)
        with open(self.persist_path, "w") as f:
            json.dump(data, f)
            
    def load_from_disk(self):
        """Load from disk if available."""
        if os.path.exists(self.persist_path):
            with open(self.persist_path, "r") as f:
                data = json.load(f)
            
            cursor = self.conn.cursor()
            cursor.executemany('''
                INSERT INTO entities (id, entity_text, entity_type, source_document, chunk_id, page_number, section)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', data.get("entities", []))
            
            cursor.executemany('''
                INSERT INTO relationships (id, source_entity, relation, target_entity, source_document)
                VALUES (?, ?, ?, ?, ?)
            ''', data.get("relationships", []))
            self.conn.commit()
            return True
        return False


class StructureAwareIngestor:
    """
    Processes documents, extracts structure, and populates the Corpus Analytics index.
    """
    def __init__(self, analytics_index: CorpusAnalytics):
        self.analytics = analytics_index
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("Spacy en_core_web_sm not found. Entity extraction will be limited.")
            self.nlp = None

    def extract_entities_spacy(self, text: str, source_doc: str, chunk_id: int):
        if not self.nlp:
            return
        
        doc = self.nlp(text)
        for ent in doc.ents:
            # We want to capture meaningful entities, not just dates/cardinals if we can avoid it,
            # but generic NER is a start. We map Spacy's labels to a more generic format.
            if ent.label_ not in ["DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "CARDINAL"]:
                 self.analytics.add_entity(ent.text, ent.label_, source_doc, chunk_id)
                 
    def extract_heuristic_relationships(self, text: str, source_doc: str):
        # Basic heuristic relationship extraction (e.g. A is a type of B)
        # "X is a Y", "X refers to Y"
        is_a_pattern = r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:is a|are|is considered|refers to)\s+([a-z]+(?:\s+[a-z]+)*)'
        matches = re.finditer(is_a_pattern, text)
        for match in matches:
            source_entity = match.group(1).strip()
            target_entity = match.group(2).strip()
            if len(source_entity) > 2 and len(target_entity) > 2:
                self.analytics.add_relationship(source_entity, "IS_A", target_entity, source_doc)

    def process_chunk(self, chunk_text: str, source_doc: str, chunk_id: int, page_num: int = None, section: str = None):
        """Process a single document chunk for analytics."""
        self.extract_entities_spacy(chunk_text, source_doc, chunk_id)
        self.extract_heuristic_relationships(chunk_text, source_doc)

    def finish(self):
        self.analytics.save_to_disk()
