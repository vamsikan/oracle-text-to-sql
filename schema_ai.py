from typing import Dict, List, Set, Tuple, Optional, Any
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BertConfig
from google.cloud import aiplatform
import numpy as np
from dataclasses import dataclass
from collections import defaultdict
import re
import os
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Available classification models
CLASSIFICATION_MODELS = {
    'deberta-v3-base': 'microsoft/deberta-v3-base',
    'deberta-v3-large': 'microsoft/deberta-v3-large',
    'deberta-v2-xlarge': 'microsoft/deberta-v2-xlarge',
    'roberta-base': 'roberta-base',
    'roberta-large': 'roberta-large',
    'bert-base': 'bert-base-uncased',
    'bert-large': 'bert-large-uncased',
    'distilbert': 'distilbert-base-uncased',
    'deberta-v3-small': 'microsoft/deberta-v3-small'
}

# Model-specific configurations
MODEL_CONFIGS = {
    'bert-base': {
        'max_length': 512,
        'batch_size': 32,
        'learning_rate': 2e-5,
        'num_labels': 2,  # Binary classification
        'hidden_dropout_prob': 0.1,
        'attention_probs_dropout_prob': 0.1
    }
}

@dataclass
class SchemaEntity:
    name: str
    type: str  # 'table', 'column', 'constraint'
    description: str
    embeddings: np.ndarray
    related_entities: Set[str]

class SchemaAI:
    def __init__(self, classification_model: str = 'bert-base'):
        # Initialize Google Cloud
        aiplatform.init(
            project=os.getenv('GOOGLE_CLOUD_PROJECT'),
            location=os.getenv('GOOGLE_CLOUD_LOCATION', 'us-central1')
        )
        
        # Initialize embedding model (Vertex AI)
        self.embedding_model = aiplatform.TextEmbeddingModel.from_pretrained(
            'textembedding-gecko@latest'
        )
        
        # Initialize classification model (BERT)
        model_path = 'bert-base-uncased' if classification_model == 'bert-base' else 'bert-large-uncased'
        config = BertConfig(
            vocab_size=30522,
            hidden_size=768 if classification_model == 'bert-base' else 1024,
            num_hidden_layers=12 if classification_model == 'bert-base' else 24,
            num_attention_heads=12 if classification_model == 'bert-base' else 16,
            intermediate_size=3072 if classification_model == 'bert-base' else 4096,
            max_position_embeddings=512,
            type_vocab_size=2,
            initializer_range=0.02
        )
        
        self.classification_model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            config=config
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        if torch.cuda.is_available():
            self.classification_model = self.classification_model.to('cuda')
            logger.info("Using CUDA for model inference")
        
        # Initialize storage
        self.entities: Dict[str, SchemaEntity] = {}
        self.entity_clusters: Dict[str, List[str]] = defaultdict(list)
        self.business_term_mappings: Dict[str, str] = {}
        
        logger.info("Successfully initialized SchemaAI")

    def get_available_models(self) -> Dict[str, str]:
        """Get list of available classification models"""
        return CLASSIFICATION_MODELS

    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a specific model"""
        if model_name not in CLASSIFICATION_MODELS:
            raise ValueError(f"Unsupported model: {model_name}")
        
        model = self.classification_model
        info = {
            'name': model_name,
            'path': CLASSIFICATION_MODELS[model_name],
            'parameters': sum(p.numel() for p in model.parameters()),
            'layers': model.config.num_hidden_layers,
            'hidden_size': model.config.hidden_size,
            'attention_heads': model.config.num_attention_heads
        }
        
        # Add BERT-specific information
        if model_name == 'bert-base':
            info.update({
                'vocab_size': model.config.vocab_size,
                'max_position_embeddings': model.config.max_position_embeddings,
                'type_vocab_size': model.config.type_vocab_size,
                'initializer_range': model.config.initializer_range
            })
        
        return info

    def extract_meaningful_terms(self, text: str) -> List[str]:
        """Extract meaningful terms from text using NLP"""
        # Remove special characters and split into words
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Remove common stop words and short terms
        stop_words = {'the', 'and', 'or', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        meaningful_terms = [word for word in words if word not in stop_words and len(word) > 2]
        
        return meaningful_terms

    def generate_embeddings(self, text: str) -> np.ndarray:
        """Generate embeddings using Vertex AI"""
        try:
            embeddings = self.embedding_model.get_embeddings([text])[0]
            return np.array(embeddings.values)
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            return np.zeros(768)  # Default dimension for text embeddings

    def analyze_schema(self, schema_data: Dict) -> None:
        """Analyze schema and build entity relationships"""
        # Process tables
        for table_name, table_info in schema_data['tables'].items():
            # Generate table description
            table_desc = f"{table_name} {table_info.get('description', '')}"
            table_terms = self.extract_meaningful_terms(table_desc)
            
            # Create table entity
            self.entities[table_name] = SchemaEntity(
                name=table_name,
                type='table',
                description=table_desc,
                embeddings=self.generate_embeddings(table_desc),
                related_entities=set()
            )
            
            # Process columns
            for column in table_info['columns']:
                col_name = column['name']
                col_desc = f"{col_name} {column.get('description', '')}"
                col_terms = self.extract_meaningful_terms(col_desc)
                
                # Create column entity
                self.entities[col_name] = SchemaEntity(
                    name=col_name,
                    type='column',
                    description=col_desc,
                    embeddings=self.generate_embeddings(col_desc),
                    related_entities={table_name}
                )
                
                # Update table's related entities
                self.entities[table_name].related_entities.add(col_name)

    def cluster_entities(self, similarity_threshold: float = 0.7) -> None:
        """Cluster similar entities using embeddings"""
        entity_names = list(self.entities.keys())
        
        for i, name1 in enumerate(entity_names):
            for name2 in entity_names[i+1:]:
                entity1 = self.entities[name1]
                entity2 = self.entities[name2]
                
                # Calculate cosine similarity
                similarity = np.dot(entity1.embeddings, entity2.embeddings) / (
                    np.linalg.norm(entity1.embeddings) * np.linalg.norm(entity2.embeddings)
                )
                
                if similarity > similarity_threshold:
                    # Add to same cluster
                    cluster_id = f"cluster_{len(self.entity_clusters)}"
                    self.entity_clusters[cluster_id].extend([name1, name2])

    def generate_business_terms(self) -> Dict[str, Dict]:
        """Generate business terms from entity clusters"""
        business_terms = {}
        
        for cluster_id, entities in self.entity_clusters.items():
            # Get all descriptions in the cluster
            descriptions = [self.entities[entity].description for entity in entities]
            
            # Generate a business term from the most common meaningful terms
            common_terms = self.extract_meaningful_terms(' '.join(descriptions))
            term_freq = defaultdict(int)
            for term in common_terms:
                term_freq[term] += 1
            
            # Get the most frequent terms
            most_common = sorted(term_freq.items(), key=lambda x: x[1], reverse=True)[:3]
            business_term = ' '.join(term for term, _ in most_common)
            
            # Create business term mapping
            business_terms[business_term] = {
                'term': business_term,
                'abbreviations': self.generate_abbreviations(business_term),
                'description': self.generate_description(descriptions),
                'related_tables': [e for e in entities if self.entities[e].type == 'table'],
                'related_columns': self.get_related_columns(entities)
            }
            
        return business_terms

    def generate_abbreviations(self, term: str) -> List[str]:
        """Generate possible abbreviations for a term"""
        words = term.split()
        abbreviations = []
        
        # Generate acronym
        acronym = ''.join(word[0] for word in words)
        abbreviations.append(acronym.lower())
        
        # Generate common variations
        if len(words) > 1:
            abbreviations.append(''.join(word[:2] for word in words).lower())
            abbreviations.append(''.join(word[0] + word[-1] for word in words).lower())
        
        return abbreviations

    def generate_description(self, descriptions: List[str]) -> str:
        """Generate a concise description from multiple entity descriptions"""
        # Combine all descriptions
        combined_desc = ' '.join(descriptions)
        
        # BERT-specific tokenization
        if self.current_model_name == 'bert-base':
            inputs = self.tokenizer(
                combined_desc,
                return_tensors="pt",
                truncation=True,
                max_length=MODEL_CONFIGS['bert-base']['max_length'],
                padding='max_length'
            )
        else:
            inputs = self.tokenizer(combined_desc, return_tensors="pt", truncation=True, max_length=512)
        
        outputs = self.classification_model(**inputs)
        
        # Get the most important words
        important_words = self.extract_meaningful_terms(combined_desc)
        
        # Create a concise description
        return f"Entity related to {' '.join(important_words[:5])}"

    def get_related_columns(self, entities: List[str]) -> Dict[str, List[str]]:
        """Get columns related to each table in the cluster"""
        related_columns = defaultdict(list)
        
        for entity in entities:
            if self.entities[entity].type == 'table':
                for related in self.entities[entity].related_entities:
                    if self.entities[related].type == 'column':
                        related_columns[entity].append(related)
        
        return dict(related_columns)

    def understand_query(self, query: str) -> Dict:
        """Understand a natural language query and map it to schema entities"""
        # Generate query embedding
        query_embedding = self.generate_embeddings(query)
        
        # Find most similar entities
        similarities = {}
        for entity_name, entity in self.entities.items():
            similarity = np.dot(query_embedding, entity.embeddings) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(entity.embeddings)
            )
            similarities[entity_name] = similarity
        
        # Get top matching entities
        top_entities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Extract query terms
        query_terms = self.extract_meaningful_terms(query)
        
        return {
            'query_terms': query_terms,
            'matched_entities': [entity for entity, _ in top_entities],
            'entity_similarities': dict(top_entities)
        }

    def save_analysis(self, filepath: str):
        """Save the AI analysis results"""
        analysis_data = {
            'entities': {
                name: {
                    'type': entity.type,
                    'description': entity.description,
                    'related_entities': list(entity.related_entities)
                }
                for name, entity in self.entities.items()
            },
            'clusters': dict(self.entity_clusters),
            'business_terms': self.generate_business_terms()
        }
        
        with open(filepath, 'w') as f:
            json.dump(analysis_data, f, indent=2) 