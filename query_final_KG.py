# RAG USING KNOWLEDGE GRAPH WITH ENHANCED ENTITY EXTRACTION AND MULTI-FILE SUPPORT
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

# Import preprocessing functions
import pdfplumber
import docx
import email
from email import policy
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_KEY")

try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
except ImportError as e:
    print(f"Error importing sentence_transformers: {e}")
    print("Try: pip install protobuf==3.20.3")
    sys.exit(1)

# ===================== PREPROCESSING FUNCTIONS =====================

def chunk_text(text, chunk_size=750, chunk_overlap=200):
    """Chunk text using RecursiveCharacterTextSplitter"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

def read_pdf(path):
    """Extract text from PDF file"""
    with pdfplumber.open(path) as pdf:
        return "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())

def read_docx(path):
    """Extract text from DOCX file"""
    doc = docx.Document(path)
    return "\n".join([para.text for para in doc.paragraphs])

def read_email(path):
    """Extract text from EML email file"""
    with open(path, 'r', encoding='utf-8') as f:
        msg = email.message_from_file(f, policy=policy.default)
        if msg.is_multipart():
            parts = [part.get_payload(decode=True).decode(errors='ignore')
                     for part in msg.walk()
                     if part.get_content_type() == 'text/plain']
            return "\n".join(parts)
        return msg.get_payload(decode=True).decode(errors='ignore')

def read_txt(path):
    """Extract text from TXT file"""
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def load_text_from_folder(data_folder="data"):
    """Load and extract text from all supported files in the data folder"""
    all_texts = []
    supported_extensions = {'.pdf', '.docx', '.eml', '.txt'}
    
    if not os.path.exists(data_folder):
        console.print(f"[red]Error: Data folder '{data_folder}' does not exist![/]")
        return all_texts
    
    console.print(f"[blue]Loading files from '{data_folder}'...[/]")
    
    for filename in os.listdir(data_folder):
        full_path = os.path.join(data_folder, filename)
        
        if not os.path.isfile(full_path):
            continue
            
        file_ext = os.path.splitext(filename)[1].lower()
        
        if file_ext not in supported_extensions:
            continue
        
        try:
            if file_ext == '.pdf':
                text = read_pdf(full_path)
            elif file_ext == '.docx':
                text = read_docx(full_path)
            elif file_ext == '.eml':
                text = read_email(full_path)
            elif file_ext == '.txt':
                text = read_txt(full_path)
            else:
                continue
            
            if text and text.strip():
                all_texts.append((filename, text))
                console.print(f"  [green]✓ Loaded: {filename}[/]")
            else:
                console.print(f"  [yellow]⚠ Empty file: {filename}[/]")
                
        except Exception as e:
            console.print(f"  [red]✗ Error loading {filename}: {e}[/]")
    
    console.print(f"[green]Successfully loaded {len(all_texts)} files[/]")
    return all_texts

def build_faiss_index(doc_chunks, embedder):
    """Build FAISS index from document chunks"""
    console.print("[blue]Building FAISS index...[/]")
    embeddings = embedder.encode(doc_chunks, show_progress_bar=True, convert_to_tensor=True, normalize_embeddings=True)
    
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.cpu().numpy())
    
    return index

def process_and_index_data(data_folder="data", force_rebuild=False):
    """Process all files in data folder and build/load FAISS index"""
    faiss_dir = "faiss_index"
    index_path = os.path.join(faiss_dir, "index.faiss")
    chunks_path = os.path.join(faiss_dir, "chunks.pkl")
    
    # Check if index already exists
    if os.path.exists(index_path) and os.path.exists(chunks_path) and not force_rebuild:
        console.print("[green]Loading existing FAISS index...[/]")
        try:
            index = faiss.read_index(index_path)
            with open(chunks_path, "rb") as f:
                doc_chunks, sources = pickle.load(f)
            console.print(f"[green]Loaded {len(doc_chunks)} chunks from disk[/]")
            return index, doc_chunks, sources
        except Exception as e:
            console.print(f"[yellow]Failed to load existing index: {e}, rebuilding...[/]")
    
    # Load embedder
    embedder = SentenceTransformer("./all-MiniLM-L6-v2", device="cpu")
    
    # Load and process all files
    all_texts = load_text_from_folder(data_folder)
    
    if not all_texts:
        console.print("[red]No files found in data folder! Please add documents to process.[/]")
        sys.exit(1)
    
    doc_chunks = []
    sources = []
    
    console.print("[blue]Chunking documents...[/]")
    for filename, text in all_texts:
        chunks = chunk_text(text)
        doc_chunks.extend(chunks)
        sources.extend([filename] * len(chunks))
    
    console.print(f"[green]Created {len(doc_chunks)} chunks from {len(all_texts)} files[/]")
    
    # Build FAISS index
    index = build_faiss_index(doc_chunks, embedder)
    
    # Save to disk
    os.makedirs(faiss_dir, exist_ok=True)
    faiss.write_index(index, index_path)
    
    with open(chunks_path, "wb") as f:
        pickle.dump((doc_chunks, sources), f)
    
    console.print("[green]FAISS index built and saved successfully![/]")
    
    return index, doc_chunks, sources

def check_if_rebuild_needed(data_folder="data"):
    """Check if rebuild is needed by comparing file metadata"""
    metadata_path = "faiss_index/file_metadata.pkl"
    
    # Get current files in data folder
    current_files = {}
    if os.path.exists(data_folder):
        for filename in os.listdir(data_folder):
            full_path = os.path.join(data_folder, filename)
            if os.path.isfile(full_path):
                # Store filename and modification time
                current_files[filename] = os.path.getmtime(full_path)
    
    # Check if metadata exists
    if not os.path.exists(metadata_path):
        # Save current metadata
        os.makedirs("faiss_index", exist_ok=True)
        with open(metadata_path, "wb") as f:
            pickle.dump(current_files, f)
        console.print("[yellow]No existing index found. Building from scratch...[/]")
        return True
    
    # Load previous metadata
    try:
        with open(metadata_path, "rb") as f:
            previous_files = pickle.load(f)
    except:
        console.print("[yellow]Could not load file metadata. Rebuilding...[/]")
        return True
    
    # Compare files
    if set(current_files.keys()) != set(previous_files.keys()):
        console.print(f"[yellow]File count changed: {len(previous_files)} → {len(current_files)} files. Rebuilding...[/]")
        # Update metadata
        with open(metadata_path, "wb") as f:
            pickle.dump(current_files, f)
        return True
    
    # Check if any files were modified
    for filename, mtime in current_files.items():
        if filename not in previous_files or previous_files[filename] != mtime:
            console.print(f"[yellow]File '{filename}' was added or modified. Rebuilding...[/]")
            # Update metadata
            with open(metadata_path, "wb") as f:
                pickle.dump(current_files, f)
            return True
    
    console.print("[green]All files unchanged. Using cached index.[/]")
    return False

# ===================== LOAD MODELS AND DATA =====================

try:
    embedder = SentenceTransformer("./all-MiniLM-L6-v2")
    cross_encoder = CrossEncoder('./local_cross_encoder', device='cpu')
    
    # Check if we need to rebuild by comparing file counts
    console.print("[blue]Checking for changes in data folder...[/]")
    rebuild_needed = check_if_rebuild_needed("data")
    
    # Process data folder and load/build index
    index, doc_chunks, sources = process_and_index_data(data_folder="data", force_rebuild=rebuild_needed)
    
except Exception as e:
    console.print(f"[red]Error loading models or data: {e}[/]")
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
        
        # 1. Named entities using spaCy
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
        
        # 4. Policy terms
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
        
        # Deduplicate entities
        unique_entities = {}
        for entity in entities:
            key = entity.name.lower()
            if key not in unique_entities or unique_entities[key].confidence < entity.confidence:
                unique_entities[key] = entity
        
        return list(unique_entities.values())
    
    def _map_spacy_label(self, label: str) -> EntityType:
        """Map spaCy entity labels to custom EntityType"""
        mapping = {
            "MONEY": EntityType.MONETARY,
            "DATE": EntityType.TEMPORAL,
            "TIME": EntityType.TEMPORAL,
            "ORG": EntityType.ORGANIZATION,
            "PERSON": EntityType.PERSON,
            "GPE": EntityType.LOCATION,
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
        self.graph = nx.MultiDiGraph()
        self.entity_extractor = EnhancedEntityExtractor()
        self.entity_index = {}
        self.chunk_entities = {}
        
    def build_graph(self, force_rebuild: bool = False) -> nx.MultiDiGraph:
        """Build enhanced knowledge graph"""
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
        
        # Add chunk nodes
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
        
        # Extract entities
        all_entities = {}
        
        for idx, chunk in enumerate(self.doc_chunks):
            chunk_entities = self.entity_extractor.extract_entities(chunk)
            self.chunk_entities[idx] = chunk_entities
            
            for entity in chunk_entities:
                if entity.name not in all_entities:
                    all_entities[entity.name] = entity
                else:
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
                mention_count=0
            )
        
        # Add relationships
        for idx, chunk_entities in self.chunk_entities.items():
            chunk_id = f"chunk_{idx}"
            
            for entity in chunk_entities:
                entity_id = self.entity_index[entity.name]
                
                self.graph.add_edge(
                    chunk_id,
                    entity_id,
                    type=RelationType.MENTIONS.value,
                    confidence=entity.confidence,
                    chunk_idx=idx
                )
                
                self.graph.nodes[entity_id]['mention_count'] += 1
        
        self._add_semantic_relationships()
        self._add_cooccurrence_relationships()
        self._calculate_graph_metrics()
        
        # Save to pickle
        data = {
            'graph': self.graph,
            'entity_index': self.entity_index,
            'chunk_entities': self.chunk_entities
        }
        with open(kg_pickle, "wb") as f:
            pickle.dump(data, f)
        
        console.print("[green]Enhanced KG built and saved![/]")
        return self.graph
    
    def _add_semantic_relationships(self):
        """Add semantic relationships between entities"""
        entity_nodes = [(n, d) for n, d in self.graph.nodes(data=True) if d['type'] == 'entity']
        
        for i, (ent1_id, ent1_data) in enumerate(entity_nodes):
            for j, (ent2_id, ent2_data) in enumerate(entity_nodes[i+1:], i+1):
                
                if (ent1_data['entity_type'] == EntityType.POLICY_TERM.value and 
                    ent2_data['entity_type'] == EntityType.COVERAGE.value):
                    
                    common_chunks = self._find_common_chunks(ent1_id, ent2_id)
                    if common_chunks:
                        relation_type = self._infer_relationship(ent1_data['name'], ent2_data['name'])
                        self.graph.add_edge(
                            ent1_id, ent2_id,
                            type=relation_type,
                            confidence=0.7,
                            evidence_chunks=common_chunks
                        )
                
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
        """Add co-occurrence relationships"""
        for idx, chunk_entities in self.chunk_entities.items():
            if len(chunk_entities) < 2:
                continue
            
            for i, ent1 in enumerate(chunk_entities):
                for ent2 in chunk_entities[i+1:]:
                    ent1_id = self.entity_index[ent1.name]
                    ent2_id = self.entity_index[ent2.name]
                    
                    if self.graph.has_edge(ent1_id, ent2_id):
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
        """Find chunks where both entities appear"""
        chunks1 = {self.graph[n][ent1_id][0]['chunk_idx'] 
                  for n in self.graph.predecessors(ent1_id) 
                  if self.graph.nodes[n]['type'] == 'chunk'}
        chunks2 = {self.graph[n][ent2_id][0]['chunk_idx'] 
                  for n in self.graph.predecessors(ent2_id) 
                  if self.graph.nodes[n]['type'] == 'chunk'}
        return list(chunks1.intersection(chunks2))
    
    def _infer_relationship(self, ent1_name: str, ent2_name: str) -> str:
        """Infer relationship type"""
        exclusion_indicators = ['not covered', 'excluded', 'not applicable']
        coverage_indicators = ['covered', 'included', 'applicable']
        
        if any(word in ent1_name.lower() for word in exclusion_indicators):
            return RelationType.EXCLUDES.value
        elif any(word in ent1_name.lower() for word in coverage_indicators):
            return RelationType.COVERS.value
        else:
            return RelationType.APPLIES_TO.value
    
    def _calculate_graph_metrics(self):
        """Calculate centrality metrics"""
        entity_subgraph = self.graph.subgraph([n for n, d in self.graph.nodes(data=True) 
                                             if d['type'] == 'entity'])
        
        if len(entity_subgraph.nodes()) > 1:
            undirected = entity_subgraph.to_undirected()
            
            try:
                centrality = nx.degree_centrality(undirected)
                betweenness = nx.betweenness_centrality(undirected)
                
                for node_id in entity_subgraph.nodes():
                    self.graph.nodes[node_id]['degree_centrality'] = centrality.get(node_id, 0)
                    self.graph.nodes[node_id]['betweenness_centrality'] = betweenness.get(node_id, 0)
            except:
                for node_id in entity_subgraph.nodes():
                    self.graph.nodes[node_id]['degree_centrality'] = 0
                    self.graph.nodes[node_id]['betweenness_centrality'] = 0
    
    def query_graph(self, query_entities: Set[str], max_hops: int = 2, max_chunks: int = 15) -> List[int]:
        """Query graph for relevant chunks"""
        retrieved_chunks = set()
        
        matched_entities = self._match_query_entities(query_entities)
        
        for entity_id in matched_entities:
            for pred in self.graph.predecessors(entity_id):
                if self.graph.nodes[pred]['type'] == 'chunk':
                    retrieved_chunks.add(self.graph.nodes[pred]['idx'])
        
        for entity_id in matched_entities:
            for successor in self.graph.successors(entity_id):
                if self.graph.nodes[successor]['type'] == 'entity':
                    edge_data = list(self.graph[entity_id][successor].values())[0]
                    if (edge_data.get('type') in [RelationType.COVERS.value, RelationType.APPLIES_TO.value] 
                        and edge_data.get('confidence', 0) > 0.6):
                        
                        for pred in self.graph.predecessors(successor):
                            if self.graph.nodes[pred]['type'] == 'chunk':
                                retrieved_chunks.add(self.graph.nodes[pred]['idx'])
        
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
        
        result = sorted(list(retrieved_chunks))[:max_chunks]
        return result
    
    def _match_query_entities(self, query_entities: Set[str]) -> Set[str]:
        """Match query entities to graph entities"""
        matched = set()
        
        for query_ent in query_entities:
            query_ent_lower = query_ent.lower()
            
            if query_ent_lower in self.entity_index:
                matched.add(self.entity_index[query_ent_lower])
                continue
            
            for entity_id in self.entity_index.values():
                entity_data = self.graph.nodes[entity_id]
                if any(query_ent_lower in alias.lower() for alias in entity_data.get('aliases', [])):
                    matched.add(entity_id)
                    continue
            
            for entity_name, entity_id in self.entity_index.items():
                if query_ent_lower in entity_name or entity_name in query_ent_lower:
                    matched.add(entity_id)
        
        return matched
    
    def get_graph_stats(self) -> Dict:
        """Get graph statistics"""
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

# --------------------- Groq LLM wrapper ---------------------
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
# Force rebuild KG if index was rebuilt
KG = enhanced_kg.build_graph(force_rebuild=rebuild_needed)

# Print graph statistics
stats = enhanced_kg.get_graph_stats()
console.print(f"[green]KG Stats: {stats['total_nodes']} nodes, {stats['total_edges']} edges[/]")
console.print(f"[cyan]Entity types: {stats['entity_types']}[/]")

# --------------------- Enhanced Query Pipeline ---------------------

def extract_specific_facts(chunks, user_query, query_entities_objects):
    """Extract specific coverage facts from document chunks"""
    facts = []
    monetary_pattern = r'(?:inr|rs\.?|₹|\$)\s?[\d,]+(?:\.\d{2})?(?:\s?(?:lakhs?|crores?|thousands?))?'
    temporal_pattern = r'\b\d+\s+(?:months?|days?|years?|weeks?)\b'
    exclusion_pattern = r'\b(excluded|not covered|not applicable|waiting period)\b'
    coverage_pattern = r'\b(covered|included|available|applicable)\b'
    
    entity_names = {ent.name for ent in query_entities_objects}
    
    for chunk in chunks:
        chunk_lower = chunk.lower()
        
        for match in re.findall(monetary_pattern, chunk_lower, flags=re.I):
            facts.append(f"Monetary: {match.strip()}")
        
        for match in re.findall(temporal_pattern, chunk_lower, flags=re.I):
            facts.append(f"Time period: {match.strip()}")
        
        for match in re.findall(exclusion_pattern, chunk_lower, flags=re.I):
            facts.append(f"Exclusion: {match.strip()}")
        
        for match in re.findall(coverage_pattern, chunk_lower, flags=re.I):
            facts.append(f"Coverage: {match.strip()}")
        
        for ent_name in entity_names:
            if ent_name in chunk_lower:
                facts.append(f"Entity mentioned: {ent_name}")
    
    return list(dict.fromkeys(facts))

def enhanced_query_pipeline(user_query, top_k=15, rerank_k=8, kg_hops=2, kg_max_chunks=15):
    """Enhanced query processing pipeline with improved Graph-RAG"""
    try:
        refined_query = reformulate_query(user_query)
        console.print(f"[dim]Refined query: {refined_query}[/]")

        # Vector retrieval with increased top_k
        query_embedding = embedder.encode([refined_query], convert_to_tensor=False, normalize_embeddings=True)
        distance, indices = index.search(query_embedding, top_k)
        retrieved_indices = [int(i) for i in indices[0] if i != -1]
        retrieved_chunks = [doc_chunks[i] for i in retrieved_indices]
        retrieved_sources = [sources[i] for i in retrieved_indices]
        
        console.print(f"[dim]Vector search found {len(retrieved_indices)} candidates from: {set(retrieved_sources)}[/]")

        # Enhanced KG retrieval
        query_entities_objects = enhanced_kg.entity_extractor.extract_entities(refined_query)
        query_entities = {ent.name for ent in query_entities_objects}
        
        query_tokens = [t.text.lower() for t in nlp(refined_query) 
                       if not t.is_stop and not t.is_punct and len(t.text) > 2]
        query_entities.update(query_tokens)
        
        console.print(f"[dim]Extracted entities: {list(query_entities)[:10]}[/]")
        
        kg_indices = enhanced_kg.query_graph(query_entities, max_hops=kg_hops, max_chunks=kg_max_chunks)
        kg_chunks = [doc_chunks[i] for i in kg_indices]
        kg_sources = [sources[i] for i in kg_indices]
        
        console.print(f"[dim]KG search found {len(kg_indices)} candidates from: {set(kg_sources)}[/]")

        # Intelligent combination - prioritize diversity
        combined_idx_order = []
        seen_sources = set()
        
        # First pass: Add one chunk from each unique source (diversity)
        for i in kg_indices + retrieved_indices:
            if i not in combined_idx_order:
                src = sources[i]
                if src not in seen_sources:
                    combined_idx_order.append(i)
                    seen_sources.add(src)
        
        # Second pass: Add remaining chunks
        for i in kg_indices + retrieved_indices:
            if i not in combined_idx_order:
                combined_idx_order.append(i)

        combined_chunks = [doc_chunks[i] for i in combined_idx_order]
        combined_sources = [sources[i] for i in combined_idx_order]

        if len(combined_chunks) == 0:
            combined_chunks = retrieved_chunks
            combined_sources = retrieved_sources
            combined_idx_order = retrieved_indices

        rerank_candidates = combined_chunks[:top_k]
        rerank_sources = combined_sources[:top_k]

        # Cross-encoder reranking
        pairs = [(refined_query, chunk) for chunk in rerank_candidates]
        with torch.no_grad():
            scores = cross_encoder.predict(pairs, batch_size=8, convert_to_numpy=True)

        reranked = sorted(
            zip(rerank_candidates, scores, combined_idx_order[:len(rerank_candidates)], rerank_sources), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        console.print(f"[dim]Top reranked sources: {[src for _, _, _, src in reranked[:5]]}[/]")
        
        top_reranked = reranked[:rerank_k]
        top_chunks = [c for c, s, idx, src in top_reranked]
        top_chunk_indices = [idx for c, s, idx, src in top_reranked]
        top_chunk_sources = [src for c, s, idx, src in top_reranked]

        # Enhanced answer generation
        matched_entity_info = []
        if query_entities_objects:
            matched_entity_info = [f"{ent.name} ({ent.entity_type.value})" 
                                 for ent in query_entities_objects[:5]]

        specific_facts = extract_specific_facts(top_chunks, user_query, query_entities_objects)
        
        answer_prompt = f"""
You are an expert document analyst who provides SPECIFIC, DEFINITIVE answers from the provided documents. You MUST NOT give vague or generic responses.

CRITICAL INSTRUCTIONS:
1. Extract EXACT information (details, amounts, dates, policies, procedures) from the provided document snippets
2. If the documents contain specific information, state it definitively (e.g., "Data is retained for 2 years" NOT "data may be retained")
3. Quote specific values, percentages, time periods, and conditions directly from the documents
4. If information is missing, say "The provided documents do not specify [specific detail]"
5. DO NOT use phrases like "generally", "typically", "may have", "could apply" - be definitive based on what's in the documents
6. Focus on the EXACT question asked - don't provide general information or make assumptions
7. Answer based on whatever type of document is provided (policy, privacy document, technical document, etc.) - do NOT assume all documents are insurance policies

User Question: "{user_query}"

Key Facts Extracted:
{chr(10).join([f"• {fact}" for fact in specific_facts]) if specific_facts else "• No specific facts extracted"}

Document Excerpts:
{chr(10).join([f'"{chunk.strip()}" [Source: {src}]' for chunk, src in zip(top_chunks, top_chunk_sources)])}

Provide a specific, definitive answer based ONLY on the exact information in these document excerpts. If the excerpts don't contain the specific information needed, clearly state what is missing:"""

        final_answer_text = cached_llm_call(answer_prompt)
        
        # Enhanced provenance
        kg_source_count = len([i for i in top_chunk_indices if i in kg_indices])
        vector_source_count = len(top_chunk_indices) - kg_source_count
        
        provenance_info = []
        if kg_source_count > 0:
            provenance_info.append(f"KG: {kg_source_count} sections")
        if vector_source_count > 0:
            provenance_info.append(f"Vector: {vector_source_count} sections")
        
        # Show unique sources
        unique_sources = list(dict.fromkeys(top_chunk_sources))
        provenance = f"Sources: {', '.join(unique_sources)} ({', '.join(provenance_info)})"
        
        return f"{final_answer_text}\n\n{provenance}"
        
    except Exception as e:
        print(f"Error in enhanced query pipeline: {e}")
        import traceback
        traceback.print_exc()
        return "Error: Failed to generate response."

# --------------------- Utility functions ---------------------

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
    
    entity_centrality.sort(key=lambda x: (x[2], x[3]), reverse=True)
    
    console.print(f"[bold blue]Top {top_k} Most Central Entities:[/]")
    for i, (name, etype, centrality, mentions) in enumerate(entity_centrality[:top_k]):
        console.print(f"{i+1:2d}. {name} ({etype}) - centrality: {centrality:.3f}, mentions: {mentions}")
    
    return entity_centrality[:top_k]

def rebuild_index_and_kg():
    """Force rebuild of both FAISS index and Knowledge Graph"""
    console.print("[yellow]Rebuilding FAISS index and Knowledge Graph...[/]")
    
    global index, doc_chunks, sources, enhanced_kg, KG
    
    # Rebuild FAISS index
    index, doc_chunks, sources = process_and_index_data(data_folder="data", force_rebuild=True)
    
    # Rebuild Knowledge Graph
    enhanced_kg = EnhancedKnowledgeGraph(doc_chunks, sources)
    KG = enhanced_kg.build_graph(force_rebuild=True)
    
    stats = enhanced_kg.get_graph_stats()
    console.print(f"[green]✓ Rebuilt successfully![/]")
    console.print(f"[green]KG Stats: {stats['total_nodes']} nodes, {stats['total_edges']} edges[/]")
    console.print(f"[cyan]Entity types: {stats['entity_types']}[/]")

# --------------------- Enhanced CLI test loop ---------------------
if __name__ == "__main__":
    console.print("[bold magenta]Welcome to the Enhanced Graph-augmented RAG Insurance Query System![/]")
    console.print("[green]Available commands:[/]")
    console.print("  • Type your query to get an answer")
    console.print("  • 'analyze <query>' - Analyze entities in a query")
    console.print("  • 'explore <entity>' - Explore entity neighborhood")
    console.print("  • 'central' - Show most central entities")
    console.print("  • 'stats' - Show graph statistics")
    console.print("  • 'rebuild' - Rebuild FAISS index and Knowledge Graph from data folder")
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
        
        elif user_input.lower() == 'rebuild':
            rebuild_index_and_kg()
            continue
        
        # Regular query processing
        try:
            result = enhanced_query_pipeline(user_input)
            console.print(f"\n[bold green]Answer:[/]\n{result}\n")
        except Exception as e:
            console.print(f"[red]Error processing query: {e}[/]")
            import traceback
            traceback.print_exc()