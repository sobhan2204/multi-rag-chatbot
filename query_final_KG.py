# RAG USING KNOWLEDGE GRAPH WITH ENHANCED ENTITY EXTRACTION AND RELATIONSHIP HANDLING
import os
import sys

from rich.console import Console
console = Console()

os.environ["SENTENCE_TRANSFORMERS_BACKEND"] = "torch"
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF_IMPORT"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

sys.modules['tensorflow'] = None
sys.modules['tensorflow.keras'] = None

import re
import faiss
import json
import pickle
import torch
from functools import lru_cache
import requests
from dotenv import load_dotenv
from collections import defaultdict, Counter
from typing import List, Dict, Set, Tuple, Optional
import numpy as np

import spacy
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    console.print("[red]spaCy model not found. Run: python -m spacy download en_core_web_sm[/]")
    raise

# Enhanced imports for better KG & NER handling
import networkx as nx
from dataclasses import dataclass
from enum import Enum

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_KEY")

try: # sentence transformers import karne me kuch dikkat aa rahi this issiliye try except lagaya hai
    from sentence_transformers import SentenceTransformer, CrossEncoder
except ImportError as e:
    print(f"Error importing sentence_transformers: {e}")
    print("Try: pip install protobuf==3.20.3")
    sys.exit(1)

try:
    embedder = SentenceTransformer("./all-MiniLM-L6-v2")
    cross_encoder = CrossEncoder('./local_cross_encoder', device='cpu')
    
    index = faiss.read_index("faiss_index/index.faiss")
    with open("faiss_index/chunks.pkl", "rb") as f:
        doc_chunks, sources = pickle.load(f)
except Exception as e:
    print(f"Error loading models or data: {e}")
    sys.exit(1)

# ---------------------  Knowledge Graph Framework ---------------------

class EntityType(Enum):
    CONDITION = "medical_condition"
    COVERAGE = "coverage_type"
    BENEFIT = "benefit"
    LIMIT = "limit"
    MONETARY = "monetary_amount"
    TEMPORAL = "time_period"
    LOCATION = "location"
    ORGANIZATION = "organization"
    PERSON = "person"
    PROCEDURE = "medical_procedure"
    POLICY_TERM = "policy_term"
    UNKNOWN = "unknown"

class RelationType(Enum):
    MENTIONS = "mentions"
    COVERS = "covers"
    EXCLUDES = "excludes"
    LIMITS = "limits"
    REQUIRES = "requires"
    APPLIES_TO = "applies_to"
    CO_OCCURS = "co_occurs"
    TEMPORAL_RELATION = "temporal_relation"
    PART_OF = "part_of"

@dataclass
class Entity:
    name: str
    entity_type: EntityType
    confidence: float
    aliases: Set[str]
    attributes: Dict[str, str]
    
    def __hash__(self):
        return hash(self.name.lower())

@dataclass
class Relation:
    source: str
    target: str
    relation_type: RelationType
    confidence: float
    context: str

class EnhancedEntityExtractor:
    def __init__(self):
        # Enhanced patterns for insurance domain
        self.monetary_patterns = [
            r'(?:inr|rs\.?|₹)\s?[\d,]+(?:\.\d{2})?(?:\s?(?:lakhs?|crores?|thousands?))?',
            r'[\d,]+(?:\.\d{2})?\s?(?:lakhs?|crores?|thousands?)',
            r'\$\s?[\d,]+(?:\.\d{2})?'
        ]
        
        self.temporal_patterns = [
            r'\b\d+\s+(?:months?|days?|years?|weeks?)\b',
            r'\b(?:pre|post)\s+(?:hospitalisation|hospitalization)\b',
            r'\bwaiting\s+period\b',
            r'\b(?:within|after|before)\s+\d+\s+(?:months?|days?|years?)\b'
        ]
        
        self.policy_terms = {
            "pre-existing disease", "pre-existing condition", "waiting period", 
            "sum insured", "co-payment", "deductible", "cashless", "reimbursement",
            "day care procedure", "room rent limit", "air ambulance", "critical illness",
            "maternity benefit", "dental treatment", "ayush treatment", "domiciliary treatment",
            "sub-limit", "copay", "network hospital", "planned treatment", "emergency treatment"
        }
        
        self.coverage_terms = {
            "hospitalisation", "hospitalization", "outpatient", "inpatient", "surgery",
            "consultation", "diagnostic test", "pharmacy", "ambulance", "intensive care",
            "organ transplant", "cancer treatment", "cardiac procedure", "orthopedic treatment"
        }
        
        self.condition_patterns = [
            r'\b(?:diabetes|hypertension|heart\s+disease|cancer|stroke|kidney\s+disease)\b',
            r'\b(?:covid|corona)\b',
            r'\b(?:accident|injury|fracture|burn)\b'
        ]

    def extract_entities(self, text: str) -> List[Entity]:
        """Enhanced entity extraction with better typing and confidence scores"""
        entities = []
        text_lower = text.lower()
        
        # Named all the entities using spaCy
        doc = nlp(text)
        for ent in doc.ents:
            entity_type = self._map_spacy_label(ent.label_)
            confidence = 0.8 if entity_type != EntityType.UNKNOWN else 0.5
            
            entity = Entity(
                name=self._normalize_entity(ent.text),
                entity_type=entity_type,
                confidence=confidence,
                aliases={self._normalize_entity(ent.text)},
                attributes={"spacy_label": ent.label_, "start": str(ent.start_char), "end": str(ent.end_char)}
            )
            entities.append(entity)
        
        # 2. Monetary amounts
        for pattern in self.monetary_patterns:
            for match in re.finditer(pattern, text, flags=re.I):
                entity = Entity(
                    name=self._normalize_entity(match.group()),
                    entity_type=EntityType.MONETARY,
                    confidence=0.9,
                    aliases={self._normalize_entity(match.group())},
                    attributes={"pattern": "monetary", "raw_match": match.group()}
                )
                entities.append(entity)
        
        # 3. Temporal expressions
        for pattern in self.temporal_patterns:
            for match in re.finditer(pattern, text, flags=re.I):
                entity = Entity(
                    name=self._normalize_entity(match.group()),
                    entity_type=EntityType.TEMPORAL,
                    confidence=0.85,
                    aliases={self._normalize_entity(match.group())},
                    attributes={"pattern": "temporal", "raw_match": match.group()}
                )
                entities.append(entity)
        
        # 4. Policy terms (exact match with context)
        for term in self.policy_terms:
            if re.search(r'\b' + re.escape(term) + r'\b', text_lower):
                entity = Entity(
                    name=term,
                    entity_type=EntityType.POLICY_TERM,
                    confidence=0.95,
                    aliases={term, self._normalize_entity(term)},
                    attributes={"category": "policy_term", "exact_match": "true"}
                )
                entities.append(entity)
        
        # 5. Coverage terms
        for term in self.coverage_terms:
            if re.search(r'\b' + re.escape(term) + r'\b', text_lower):
                entity = Entity(
                    name=term,
                    entity_type=EntityType.COVERAGE,
                    confidence=0.9,
                    aliases={term, self._normalize_entity(term)},
                    attributes={"category": "coverage", "exact_match": "true"}
                )
                entities.append(entity)
        
        # 6. Medical conditions
        for pattern in self.condition_patterns:
            for match in re.finditer(pattern, text, flags=re.I):
                entity = Entity(
                    name=self._normalize_entity(match.group()),
                    entity_type=EntityType.CONDITION,
                    confidence=0.8,
                    aliases={self._normalize_entity(match.group())},
                    attributes={"pattern": "medical_condition", "raw_match": match.group()}
                )
                entities.append(entity)
        
        # Deduplicate entities by normalized name
        unique_entities = {}
        for entity in entities:
            key = entity.name.lower()
            if key not in unique_entities or unique_entities[key].confidence < entity.confidence:
                unique_entities[key] = entity
        
        return list(unique_entities.values())
    
    def _map_spacy_label(self, label: str) -> EntityType:
        """Map spaCy entity labels to our custom EntityType"""
        mapping = {
            "MONEY": EntityType.MONETARY,
            "DATE": EntityType.TEMPORAL,
            "TIME": EntityType.TEMPORAL,
            "ORG": EntityType.ORGANIZATION,
            "PERSON": EntityType.PERSON,
            "GPE": EntityType.LOCATION,  # Geopolitical entity
            "LOC": EntityType.LOCATION,
        }
        return mapping.get(label, EntityType.UNKNOWN)
    
    def _normalize_entity(self, text: str) -> str:
        """Normalize entity text"""
        return re.sub(r'\s+', ' ', text.strip()).lower()

class EnhancedKnowledgeGraph:
    def __init__(self, doc_chunks: List[str], sources: List[str]):
        self.doc_chunks = doc_chunks
        self.sources = sources
        self.graph = nx.MultiDiGraph()  # Allow multiple edges between nodes
        self.entity_extractor = EnhancedEntityExtractor()
        self.entity_index = {}  # entity_name -> node_id mapping
        self.chunk_entities = {}  # chunk_idx -> List[Entity]
        
    def build_graph(self, force_rebuild: bool = False) -> nx.MultiDiGraph:
        """Build enhanced knowledge graph with better relationships"""
        kg_pickle = "enhanced_kg_graph.pkl"
        
        if os.path.exists(kg_pickle) and not force_rebuild:
            try:
                with open(kg_pickle, "rb") as f:
                    data = pickle.load(f)
                    self.graph = data['graph']
                    self.entity_index = data['entity_index']
                    self.chunk_entities = data['chunk_entities']
                console.print("[green]Loaded enhanced KG from disk.[/]")
                return self.graph
            except Exception as e:
                console.print(f"[yellow]Failed to load KG pickle: {e}, rebuilding...[/]")
        
        console.print("[blue]Building enhanced knowledge graph...[/]")
        
        # Step 1: Add chunk nodes
        for idx, chunk in enumerate(self.doc_chunks):
            chunk_id = f"chunk_{idx}"
            self.graph.add_node(
                chunk_id,
                type="chunk",
                text=chunk,
                source=self.sources[idx],
                idx=idx,
                length=len(chunk),
                word_count=len(chunk.split())
            )
        
        # Step 2: Extract entities and create entity nodes
        all_entities = {}  # entity_name -> Entity object
        
        for idx, chunk in enumerate(self.doc_chunks):
            chunk_entities = self.entity_extractor.extract_entities(chunk)
            self.chunk_entities[idx] = chunk_entities
            
            for entity in chunk_entities:
                if entity.name not in all_entities:
                    all_entities[entity.name] = entity
                else:
                    # Merge aliases and update confidence
                    existing = all_entities[entity.name]
                    existing.aliases.update(entity.aliases)
                    existing.confidence = max(existing.confidence, entity.confidence)
        
        # Add entity nodes
        for entity_name, entity in all_entities.items():
            entity_id = f"ent::{entity_name}"
            self.entity_index[entity_name] = entity_id
            
            self.graph.add_node(
                entity_id,
                type="entity",
                name=entity.name,
                entity_type=entity.entity_type.value,
                confidence=entity.confidence,
                aliases=list(entity.aliases),
                attributes=entity.attributes,
                mention_count=0  # Will be updated below
            )
        
        # Step 3: Add chunk-entity relationships
        for idx, chunk_entities in self.chunk_entities.items():
            chunk_id = f"chunk_{idx}"
            
            for entity in chunk_entities:
                entity_id = self.entity_index[entity.name]
                
                # Add MENTIONS relationship
                self.graph.add_edge(
                    chunk_id,
                    entity_id,
                    type=RelationType.MENTIONS.value,
                    confidence=entity.confidence,
                    chunk_idx=idx
                )
                
                # Update mention count
                self.graph.nodes[entity_id]['mention_count'] += 1
        
        # Step 4: Add semantic relationships between entities
        self._add_semantic_relationships()
        
        # Step 5: Add co-occurrence relationships
        self._add_cooccurrence_relationships()
        
        # Step 6: Calculate centrality measures
        self._calculate_graph_metrics()
        
        # Save to pickle
        data = {
            'graph': self.graph,
            'entity_index': self.entity_index,
            'chunk_entities': self.chunk_entities
        }
        with open(kg_pickle, "wb") as f:
            pickle.dump(data, f)
        
        console.print("[green]Enhanced KG built and saved to disk.[/]")
        return self.graph
    
    def _add_semantic_relationships(self):
        """Add semantic relationships between entities based on domain knowledge"""
        entity_pairs = []
        
        # Get all entity nodes
        entity_nodes = [(n, d) for n, d in self.graph.nodes(data=True) if d['type'] == 'entity']
        
        for i, (ent1_id, ent1_data) in enumerate(entity_nodes):
            for j, (ent2_id, ent2_data) in enumerate(entity_nodes[i+1:], i+1):
                
                # Policy terms and coverage relationships
                if (ent1_data['entity_type'] == EntityType.POLICY_TERM.value and 
                    ent2_data['entity_type'] == EntityType.COVERAGE.value):
                    
                    # Find chunks where both entities appear
                    common_chunks = self._find_common_chunks(ent1_id, ent2_id)
                    if common_chunks:
                        relation_type = self._infer_relationship(ent1_data['name'], ent2_data['name'])
                        self.graph.add_edge(
                            ent1_id, ent2_id,
                            type=relation_type,
                            confidence=0.7,
                            evidence_chunks=common_chunks
                        )
                
                # Monetary limits and coverage
                if (ent1_data['entity_type'] == EntityType.MONETARY.value and 
                    ent2_data['entity_type'] in [EntityType.COVERAGE.value, EntityType.POLICY_TERM.value]):
                    
                    common_chunks = self._find_common_chunks(ent1_id, ent2_id)
                    if common_chunks:
                        self.graph.add_edge(
                            ent1_id, ent2_id,
                            type=RelationType.LIMITS.value,
                            confidence=0.8,
                            evidence_chunks=common_chunks
                        )
    
    def _add_cooccurrence_relationships(self):
        """Add co-occurrence relationships between entities"""
        # For each chunk, connect entities that appear together
        for idx, chunk_entities in self.chunk_entities.items():
            if len(chunk_entities) < 2:
                continue
            
            # Create co-occurrence edges between all entity pairs in this chunk
            for i, ent1 in enumerate(chunk_entities):
                for ent2 in chunk_entities[i+1:]:
                    ent1_id = self.entity_index[ent1.name]
                    ent2_id = self.entity_index[ent2.name]
                    
                    # Check if edge already exists and increment weight
                    if self.graph.has_edge(ent1_id, ent2_id):
                        # Find existing co-occurrence edge
                        for edge_data in self.graph[ent1_id][ent2_id].values():
                            if edge_data.get('type') == RelationType.CO_OCCURS.value:
                                edge_data['weight'] = edge_data.get('weight', 1) + 1
                                edge_data['chunk_evidence'].append(idx)
                                break
                    else:
                        self.graph.add_edge(
                            ent1_id, ent2_id,
                            type=RelationType.CO_OCCURS.value,
                            weight=1,
                            chunk_evidence=[idx]
                        )
    
    def _find_common_chunks(self, ent1_id: str, ent2_id: str) -> List[int]:
        """Find chunks where both entities are mentioned"""
        chunks1 = {self.graph[n][ent1_id][0]['chunk_idx'] 
                  for n in self.graph.predecessors(ent1_id) 
                  if self.graph.nodes[n]['type'] == 'chunk'}
        chunks2 = {self.graph[n][ent2_id][0]['chunk_idx'] 
                  for n in self.graph.predecessors(ent2_id) 
                  if self.graph.nodes[n]['type'] == 'chunk'}
        return list(chunks1.intersection(chunks2))
    
    def _infer_relationship(self, ent1_name: str, ent2_name: str) -> str:
        """Infer relationship type between entities based on domain knowledge"""
        # Simple rule-based relationship inference
        exclusion_indicators = ['not covered', 'excluded', 'not applicable']
        coverage_indicators = ['covered', 'included', 'applicable']
        
        # This is a simplified version - in practice, you'd use more sophisticated NLP
        if any(word in ent1_name.lower() for word in exclusion_indicators):
            return RelationType.EXCLUDES.value
        elif any(word in ent1_name.lower() for word in coverage_indicators):
            return RelationType.COVERS.value
        else:
            return RelationType.APPLIES_TO.value
    
    def _calculate_graph_metrics(self):
        """Calculate and store graph centrality metrics"""
        # Only calculate for entity nodes
        entity_subgraph = self.graph.subgraph([n for n, d in self.graph.nodes(data=True) 
                                             if d['type'] == 'entity'])
        
        if len(entity_subgraph.nodes()) > 1:
            # Convert to undirected for centrality calculations
            undirected = entity_subgraph.to_undirected()
            
            try:
                centrality = nx.degree_centrality(undirected)
                betweenness = nx.betweenness_centrality(undirected)
                
                for node_id in entity_subgraph.nodes():
                    self.graph.nodes[node_id]['degree_centrality'] = centrality.get(node_id, 0)
                    self.graph.nodes[node_id]['betweenness_centrality'] = betweenness.get(node_id, 0)
            except:
                # Fallback if centrality calculation fails
                for node_id in entity_subgraph.nodes():
                    self.graph.nodes[node_id]['degree_centrality'] = 0
                    self.graph.nodes[node_id]['betweenness_centrality'] = 0
    
    def query_graph(self, query_entities: Set[str], max_hops: int = 2, max_chunks: int = 15) -> List[int]:
        """Enhanced graph querying with multiple strategies"""
        retrieved_chunks = set()
        
        # Strategy 1: Direct entity matches
        matched_entities = self._match_query_entities(query_entities)
        
        for entity_id in matched_entities:
            # Get chunks mentioning this entity
            for pred in self.graph.predecessors(entity_id):
                if self.graph.nodes[pred]['type'] == 'chunk':
                    retrieved_chunks.add(self.graph.nodes[pred]['idx'])
        
        # Strategy 2: Semantic relationship traversal
        for entity_id in matched_entities:
            # Find related entities through semantic relationships
            for successor in self.graph.successors(entity_id):
                if self.graph.nodes[successor]['type'] == 'entity':
                    # Check relationship type and confidence
                    edge_data = list(self.graph[entity_id][successor].values())[0]
                    if (edge_data.get('type') in [RelationType.COVERS.value, RelationType.APPLIES_TO.value] 
                        and edge_data.get('confidence', 0) > 0.6):
                        
                        # Get chunks for related entity
                        for pred in self.graph.predecessors(successor):
                            if self.graph.nodes[pred]['type'] == 'chunk':
                                retrieved_chunks.add(self.graph.nodes[pred]['idx'])
        
        # Strategy 3: High centrality entities (when direct matches are few)
        if len(retrieved_chunks) < 3:
            high_centrality_entities = sorted(
                [(n, d) for n, d in self.graph.nodes(data=True) 
                 if d['type'] == 'entity'],
                key=lambda x: x[1].get('degree_centrality', 0),
                reverse=True
            )[:5]
            
            for entity_id, _ in high_centrality_entities:
                if len(retrieved_chunks) >= max_chunks:
                    break
                for pred in self.graph.predecessors(entity_id):
                    if self.graph.nodes[pred]['type'] == 'chunk':
                        retrieved_chunks.add(self.graph.nodes[pred]['idx'])
        
        # Convert to sorted list and limit
        result = sorted(list(retrieved_chunks))[:max_chunks]
        return result
    
    def _match_query_entities(self, query_entities: Set[str]) -> Set[str]:
        """Match query entities to graph entities with fuzzy matching"""
        matched = set()
        
        for query_ent in query_entities:
            query_ent_lower = query_ent.lower()
            
            # Exact match
            if query_ent_lower in self.entity_index:
                matched.add(self.entity_index[query_ent_lower])
                continue
            
            # Alias match
            for entity_id in self.entity_index.values():
                entity_data = self.graph.nodes[entity_id]
                if any(query_ent_lower in alias.lower() for alias in entity_data.get('aliases', [])):
                    matched.add(entity_id)
                    continue
            
            # Substring match with entity names
            for entity_name, entity_id in self.entity_index.items():
                if query_ent_lower in entity_name or entity_name in query_ent_lower:
                    matched.add(entity_id)
        
        return matched
    
    def get_graph_stats(self) -> Dict:
        """Get statistics about the knowledge graph"""
        entity_nodes = [n for n, d in self.graph.nodes(data=True) if d['type'] == 'entity']
        chunk_nodes = [n for n, d in self.graph.nodes(data=True) if d['type'] == 'chunk']
        
        entity_types = defaultdict(int)
        for node_id in entity_nodes:
            entity_type = self.graph.nodes[node_id]['entity_type']
            entity_types[entity_type] += 1
        
        return {
            'total_nodes': len(self.graph.nodes()),
            'entity_nodes': len(entity_nodes),
            'chunk_nodes': len(chunk_nodes),
            'total_edges': len(self.graph.edges()),
            'entity_types': dict(entity_types),
            'avg_entities_per_chunk': np.mean([len(entities) for entities in self.chunk_entities.values()])
        }

# --------------------- Groq LLM wrapper (unchanged) ---------------------
def groq_call(prompt: str) -> str:
    """Make API call to Groq"""
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    payload = {
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"Error calling Groq API: {e}")
        return "Error: Unable to process request"

@lru_cache(maxsize=128)
def reformulate_query(original_query):
    """Reformulate user query for better retrieval"""
    prompt = f"""
    You are an AI assistant tasked with reformulating user queries to improve retrieval in a RAG system. 
    Given the original query, rewrite it to be more specific, detailed, and likely to retrieve relevant information.
    ALWAYS convert fragmentary notes like "34F, hotel issue Thailand" into a full question like "Does the policy cover hotel rebooking expenses in Thailand?"
    But keep the query concise and don't add any unnecessary assumptions.

    Original query: {original_query}

    Rewritten query:
    """
    return groq_call(prompt).strip()

@lru_cache(maxsize=128)
def cached_llm_call(prompt: str) -> str:
    """Cached LLM call for efficiency"""
    return groq_call(prompt).strip()

# --------------------- Initialize Enhanced KG ---------------------
enhanced_kg = EnhancedKnowledgeGraph(doc_chunks, sources)
KG = enhanced_kg.build_graph(force_rebuild=False)

# Print graph statistics
stats = enhanced_kg.get_graph_stats()
console.print(f"[green]KG Stats: {stats['total_nodes']} nodes, {stats['total_edges']} edges[/]")
console.print(f"[cyan]Entity types: {stats['entity_types']}[/]")

# --------------------- Enhanced Query Pipeline ---------------------

def enhanced_query_pipeline(user_query, top_k=10, rerank_k=5, kg_hops=2, kg_max_chunks=12):
    """Enhanced query processing pipeline with improved Graph-RAG"""
    try:
        refined_query = reformulate_query(user_query)

        # ---------------- Vector retrieval ----------------
        query_embedding = embedder.encode([refined_query], convert_to_tensor=False, normalize_embeddings=True)
        distance, indices = index.search(query_embedding, top_k)
        retrieved_indices = [int(i) for i in indices[0] if i != -1]
        retrieved_chunks = [doc_chunks[i] for i in retrieved_indices]
        retrieved_sources = [sources[i] for i in retrieved_indices]

        # ---------------- Enhanced KG retrieval ----------------
        # Extract query entities using the enhanced extractor
        query_entities_objects = enhanced_kg.entity_extractor.extract_entities(refined_query)
        query_entities = {ent.name for ent in query_entities_objects}
        
        # Add original query tokens as potential entities
        query_tokens = [t.text.lower() for t in nlp(refined_query) 
                       if not t.is_stop and not t.is_punct and len(t.text) > 2]
        query_entities.update(query_tokens)
        
        kg_indices = enhanced_kg.query_graph(query_entities, max_hops=kg_hops, max_chunks=kg_max_chunks)
        kg_chunks = [doc_chunks[i] for i in kg_indices]
        kg_sources = [sources[i] for i in kg_indices]

        # ---------------- Intelligent combination ----------------
        # Prioritize KG results for higher precision, then add vector results
        combined_idx_order = []
        
        # Add KG results first (higher precision for entity-based queries)
        for i in kg_indices:
            if i not in combined_idx_order:
                combined_idx_order.append(i)
        
        # Add vector results
        for i in retrieved_indices:
            if i not in combined_idx_order:
                combined_idx_order.append(i)

        combined_chunks = [doc_chunks[i] for i in combined_idx_order]
        combined_sources = [sources[i] for i in combined_idx_order]

        # Fallback to vector-only if no results
        if len(combined_chunks) == 0:
            combined_chunks = retrieved_chunks
            combined_sources = retrieved_sources
            combined_idx_order = retrieved_indices

        # Limit candidates for reranking
        rerank_candidates = combined_chunks[:top_k]

        # ---------------- Cross-encoder reranking ----------------
        pairs = [(refined_query, chunk) for chunk in rerank_candidates]
        with torch.no_grad():
            scores = cross_encoder.predict(pairs, batch_size=8, convert_to_numpy=True)

        reranked = sorted(
            zip(rerank_candidates, scores, combined_idx_order[:len(rerank_candidates)]), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        top_reranked = reranked[:rerank_k]
        top_chunks = [c for c, s, idx in top_reranked]
        top_chunk_indices = [idx for c, s, idx in top_reranked]
        top_chunk_sources = [sources[idx] for idx in top_chunk_indices]

        # ---------------- Enhanced answer generation ----------------
        # Include matched entities in the prompt for better context
        matched_entity_info = []
        if query_entities_objects:
            matched_entity_info = [f"{ent.name} ({ent.entity_type.value})" 
                                 for ent in query_entities_objects[:5]]

        # Extract specific facts from chunks for more targeted answers
        specific_facts = extract_specific_facts(top_chunks, user_query, query_entities_objects)
        
        answer_prompt = f"""
You are an expert insurance agent who must provide SPECIFIC, DEFINITIVE answers from policy documents. You MUST NOT give vague or generic responses.

CRITICAL INSTRUCTIONS:
1. Extract EXACT coverage details, amounts, waiting periods, exclusions from the provided snippets
2. If the documents contain specific information, state it definitively (e.g., "Coverage is available after 2 years" NOT "there may be waiting periods")
3. Quote specific amounts, percentages, time periods, and conditions directly from the policy
4. If information is missing, say "The policy documents provided do not specify [specific detail]" 
5. DO NOT use phrases like "generally", "typically", "may have", "could apply" - be definitive
6. Focus on the EXACT question asked - don't provide general information

User Question: "{user_query}"

Key Coverage Facts Extracted:
{chr(10).join([f"• {fact}" for fact in specific_facts]) if specific_facts else "• No specific facts extracted"}

Policy Document Excerpts:
{chr(10).join([f'"{chunk.strip()}" [Source: {src}]' for chunk, src in zip(top_chunks, top_chunk_sources)])}

Provide a specific, definitive answer based ONLY on the exact information in these policy excerpts. If the excerpts don't contain the specific information needed, clearly state what is missing:"""

        final_answer_text = cached_llm_call(answer_prompt)
        
        # Enhanced provenance with method indicators
        kg_source_count = len([i for i in top_chunk_indices if i in kg_indices])
        vector_source_count = len(top_chunk_indices) - kg_source_count
        
        provenance_info = []
        if kg_source_count > 0:
            provenance_info.append(f"KG: {kg_source_count} sections")
        if vector_source_count > 0:
            provenance_info.append(f"Vector: {vector_source_count} sections")
        
        provenance = f"Sources: {', '.join([str(s) for s in top_chunk_sources])} ({', '.join(provenance_info)})"
        
        return f"{final_answer_text}\n\n{provenance}"
        
    except Exception as e:
        print(f"Error in enhanced query pipeline: {e}")
        import traceback
        traceback.print_exc()
        return "Error: Failed to generate response."

# --------------------- Additional utility functions ---------------------

def extract_specific_facts(chunks, user_query, query_entities_objects):
    """
    Extract specific coverage facts (amounts, periods, exclusions, etc.) from document chunks
    using entity information and simple regex patterns.
    """
    facts = []
    # Patterns for monetary, temporal, exclusions, coverage, etc.
    monetary_pattern = r'(?:inr|rs\.?|₹|\$)\s?[\d,]+(?:\.\d{2})?(?:\s?(?:lakhs?|crores?|thousands?))?'
    temporal_pattern = r'\b\d+\s+(?:months?|days?|years?|weeks?)\b'
    exclusion_pattern = r'\b(excluded|not covered|not applicable|waiting period)\b'
    coverage_pattern = r'\b(covered|included|available|applicable)\b'
    # Combine entity names for matching
    entity_names = {ent.name for ent in query_entities_objects}
    for chunk in chunks:
        chunk_lower = chunk.lower()
        # Monetary facts
        for match in re.findall(monetary_pattern, chunk_lower, flags=re.I):
            facts.append(f"Monetary: {match.strip()}")
        # Temporal facts
        for match in re.findall(temporal_pattern, chunk_lower, flags=re.I):
            facts.append(f"Time period: {match.strip()}")
        # Exclusion facts
        for match in re.findall(exclusion_pattern, chunk_lower, flags=re.I):
            facts.append(f"Exclusion: {match.strip()}")
        # Coverage facts
        for match in re.findall(coverage_pattern, chunk_lower, flags=re.I):
            facts.append(f"Coverage: {match.strip()}")
        # Entity mentions
        for ent_name in entity_names:
            if ent_name in chunk_lower:
                facts.append(f"Entity mentioned: {ent_name}")
    # Remove duplicates and return
    return list(dict.fromkeys(facts))

def analyze_query_entities(query: str):
    """Analyze and display entities found in a query"""
    entities = enhanced_kg.entity_extractor.extract_entities(query)
    
    console.print(f"[bold blue]Query Entity Analysis for: '{query}'[/]")
    console.print(f"[green]Found {len(entities)} entities:[/]")
    
    for entity in entities:
        console.print(f"  • {entity.name} ({entity.entity_type.value}) - confidence: {entity.confidence:.2f}")
        if entity.aliases:
            console.print(f"    Aliases: {', '.join(entity.aliases)}")
    
    return entities

def explore_entity_neighborhood(entity_name: str, max_depth: int = 2):
    """Explore the neighborhood of an entity in the knowledge graph"""
    entity_name_norm = entity_name.lower()
    
    if entity_name_norm not in enhanced_kg.entity_index:
        console.print(f"[red]Entity '{entity_name}' not found in knowledge graph[/]")
        return
    
    entity_id = enhanced_kg.entity_index[entity_name_norm]
    console.print(f"[bold blue]Exploring neighborhood of: {entity_name}[/]")
    
    # Get direct connections
    successors = list(KG.successors(entity_id))
    predecessors = list(KG.predecessors(entity_id))
    
    console.print(f"[green]Direct connections ({len(successors + predecessors)}):[/]")
    
    for neighbor in successors + predecessors:
        neighbor_data = KG.nodes[neighbor]
        if neighbor_data['type'] == 'entity':
            edge_data = list(KG[entity_id][neighbor].values())[0] if neighbor in successors else list(KG[neighbor][entity_id].values())[0]
            console.print(f"  → {neighbor_data['name']} ({neighbor_data['entity_type']}) via {edge_data.get('type', 'unknown')}")
        elif neighbor_data['type'] == 'chunk':
            console.print(f"  → Document chunk {neighbor_data['idx']} (source: {neighbor_data['source']})")

def get_most_central_entities(top_k: int = 10):
    """Get the most central entities in the knowledge graph"""
    entity_centrality = []
    
    for node_id, data in KG.nodes(data=True):
        if data['type'] == 'entity':
            centrality = data.get('degree_centrality', 0)
            mention_count = data.get('mention_count', 0)
            entity_centrality.append((data['name'], data['entity_type'], centrality, mention_count))
    
    # Sort by centrality and mention count
    entity_centrality.sort(key=lambda x: (x[2], x[3]), reverse=True)
    
    console.print(f"[bold blue]Top {top_k} Most Central Entities:[/]")
    for i, (name, etype, centrality, mentions) in enumerate(entity_centrality[:top_k]):
        console.print(f"{i+1:2d}. {name} ({etype}) - centrality: {centrality:.3f}, mentions: {mentions}")
    
    return entity_centrality[:top_k]

# --------------------- Enhanced CLI test loop ---------------------
if __name__ == "__main__":
    console.print("[bold magenta]Welcome to the Enhanced Graph-augmented RAG Insurance Query System![/]")
    console.print("[green]Available commands:[/]")
    console.print("  • Type your query to get an answer")
    console.print("  • 'analyze <query>' - Analyze entities in a query")
    console.print("  • 'explore <entity>' - Explore entity neighborhood")
    console.print("  • 'central' - Show most central entities")
    console.print("  • 'stats' - Show graph statistics")
    console.print("  • 'q', 'quit', or 'exit' to leave")
    console.print()
    
    while True:
        try:
            user_input = input("Enter your query or command: ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[red]Exiting.[/]")
            break
            
        if not user_input:
            continue
            
        if user_input.lower() in ['q', 'quit', 'exit']:
            console.print("[red]THANK YOU FOR USING OUR SERVICE![/]")
            break
        
        # Handle special commands
        if user_input.startswith('analyze '):
            query_to_analyze = user_input[8:].strip()
            if query_to_analyze:
                analyze_query_entities(query_to_analyze)
            continue
        
        elif user_input.startswith('explore '):
            entity_to_explore = user_input[8:].strip()
            if entity_to_explore:
                explore_entity_neighborhood(entity_to_explore)
            continue
        
        elif user_input.lower() == 'central':
            get_most_central_entities()
            continue
        
        elif user_input.lower() == 'stats':
            stats = enhanced_kg.get_graph_stats()
            console.print("[bold blue]Knowledge Graph Statistics:[/]")
            for key, value in stats.items():
                if isinstance(value, dict):
                    console.print(f"  {key}:")
                    for k, v in value.items():
                        console.print(f"    - {k}: {v}")
                else:
                    console.print(f"  {key}: {value}")
            continue
        
        # Regular query processing
        try:
            result = enhanced_query_pipeline(user_input)
            console.print(f"\n[bold green]Answer:[/]\n{result}\n")
        except Exception as e:
            console.print(f"[red]Error processing query: {e}[/]")
            import traceback
            traceback.print_exc()