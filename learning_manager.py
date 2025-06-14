from typing import Dict, List, Set, Optional
import numpy as np
from sklearn.cluster import DBSCAN
from collections import defaultdict
import json
from datetime import datetime
import re
from dataclasses import dataclass
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BertConfig
from google.cloud import aiplatform
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import logging
from functools import lru_cache
import pickle
from pathlib import Path

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Available models configuration
LEARNING_MODELS = {
    'embedding': {
        'vertex': 'textembedding-gecko@latest'  # Vertex AI embedding model
    },
    'classification': {
        'bert-large': 'bert-large-uncased',
        'bert-base': 'bert-base-uncased'
    }
}

# Cache configuration
CACHE_DIR = Path('cache')
EMBEDDING_CACHE_SIZE = 1000
PATTERN_CACHE_SIZE = 500

@dataclass
class QueryPattern:
    pattern: str
    frequency: int
    last_used: datetime
    success_rate: float
    template: str
    parameters: List[str]

@dataclass
class DomainTerm:
    term: str
    description: str
    related_entities: Set[str]
    usage_count: int
    confidence: float

class LearningManager:
    def __init__(self):
        # Create cache directory if it doesn't exist
        CACHE_DIR.mkdir(exist_ok=True)
        
        # Initialize Google Cloud
        aiplatform.init(
            project=os.getenv('GOOGLE_CLOUD_PROJECT'),
            location=os.getenv('GOOGLE_CLOUD_LOCATION', 'us-central1')
        )
        
        # Initialize models
        self.setup_models()
        
        # Initialize storage
        self.query_patterns: Dict[str, QueryPattern] = {}
        self.domain_terms: Dict[str, DomainTerm] = {}
        self.query_templates: Dict[str, str] = {}
        self.feedback_history: List[Dict] = []
        self.uncertain_queries: List[Dict] = []
        
        # Load existing data
        self.load_data()
        
    def setup_models(self):
        """Initialize models based on configuration"""
        try:
            # Get model configurations from environment variables
            classification_model = os.getenv('LEARNING_CLASSIFICATION_MODEL', 'bert-base')
            
            # Setup embedding model (always use Vertex AI)
            self.embedding_model = aiplatform.TextEmbeddingModel.from_pretrained(
                LEARNING_MODELS['embedding']['vertex']
            )
            
            # Setup classification model (BERT)
            model_path = LEARNING_MODELS['classification'][classification_model]
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
            
            logger.info("Successfully initialized models")
        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
            raise

    @lru_cache(maxsize=EMBEDDING_CACHE_SIZE)
    def generate_embeddings(self, text: str) -> np.ndarray:
        """Generate embeddings using Vertex AI with caching"""
        try:
            embeddings = self.embedding_model.get_embeddings([text])[0]
            return np.array(embeddings.values)
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            return np.zeros(768)  # Default dimension for text embeddings

    def load_data(self):
        """Load saved learning data with error handling"""
        try:
            data_file = CACHE_DIR / 'learning_data.json'
            if data_file.exists():
                with open(data_file, 'r') as f:
                    data = json.load(f)
                    self.query_patterns = {
                        k: QueryPattern(**v) for k, v in data.get('patterns', {}).items()
                    }
                    self.domain_terms = {
                        k: DomainTerm(**v) for k, v in data.get('terms', {}).items()
                    }
                    self.query_templates = data.get('templates', {})
                    self.feedback_history = data.get('feedback', [])
                logger.info("Successfully loaded learning data")
        except Exception as e:
            logger.error(f"Error loading learning data: {str(e)}")

    def save_data(self):
        """Save learning data with error handling"""
        try:
            data = {
                'patterns': {
                    k: {
                        'pattern': v.pattern,
                        'frequency': v.frequency,
                        'last_used': v.last_used.isoformat(),
                        'success_rate': v.success_rate,
                        'template': v.template,
                        'parameters': v.parameters
                    }
                    for k, v in self.query_patterns.items()
                },
                'terms': {
                    k: {
                        'term': v.term,
                        'description': v.description,
                        'related_entities': list(v.related_entities),
                        'usage_count': v.usage_count,
                        'confidence': v.confidence
                    }
                    for k, v in self.domain_terms.items()
                },
                'templates': self.query_templates,
                'feedback': self.feedback_history
            }
            
            data_file = CACHE_DIR / 'learning_data.json'
            with open(data_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info("Successfully saved learning data")
        except Exception as e:
            logger.error(f"Error saving learning data: {str(e)}")

    @lru_cache(maxsize=PATTERN_CACHE_SIZE)
    def _extract_pattern(self, query: str) -> str:
        """Extract query pattern using NLP with caching"""
        # Tokenize and remove specific values
        tokens = self.tokenizer.tokenize(query)
        
        # Replace specific values with placeholders
        pattern_tokens = []
        for token in tokens:
            if self._is_specific_value(token):
                pattern_tokens.append('<VALUE>')
            else:
                pattern_tokens.append(token)
        
        return ' '.join(pattern_tokens)

    def analyze_feedback(self, feedback: Dict):
        """Analyze feedback for learning opportunities"""
        try:
            self.feedback_history.append(feedback)
            
            # Update query patterns
            if feedback['is_correct']:
                self._update_successful_pattern(feedback)
            else:
                self._update_failed_pattern(feedback)
            
            # Update domain terms
            self._update_domain_terms(feedback)
            
            # Update query templates
            self._update_query_templates(feedback)
            
            # Save updated data
            self.save_data()
            
            logger.info("Successfully analyzed feedback")
        except Exception as e:
            logger.error(f"Error analyzing feedback: {str(e)}")

    def _update_successful_pattern(self, feedback: Dict):
        """Update patterns based on successful queries"""
        query = feedback['query']
        sql = feedback['sql']
        
        # Extract pattern
        pattern = self._extract_pattern(query)
        
        if pattern in self.query_patterns:
            pattern_data = self.query_patterns[pattern]
            pattern_data.frequency += 1
            pattern_data.last_used = datetime.now()
            pattern_data.success_rate = (
                (pattern_data.success_rate * (pattern_data.frequency - 1) + 1) /
                pattern_data.frequency
            )
        else:
            self.query_patterns[pattern] = QueryPattern(
                pattern=pattern,
                frequency=1,
                last_used=datetime.now(),
                success_rate=1.0,
                template=self._generate_template(query, sql),
                parameters=self._extract_parameters(query)
            )

    def _update_failed_pattern(self, feedback: Dict):
        """Update patterns based on failed queries"""
        query = feedback['query']
        pattern = self._extract_pattern(query)
        
        if pattern in self.query_patterns:
            pattern_data = self.query_patterns[pattern]
            pattern_data.frequency += 1
            pattern_data.last_used = datetime.now()
            pattern_data.success_rate = (
                (pattern_data.success_rate * (pattern_data.frequency - 1)) /
                pattern_data.frequency
            )
            
            # Add to uncertain queries if success rate is low
            if pattern_data.success_rate < 0.5:
                self.uncertain_queries.append({
                    'query': query,
                    'pattern': pattern,
                    'success_rate': pattern_data.success_rate
                })

    def _update_domain_terms(self, feedback: Dict):
        """Update domain terminology based on feedback"""
        query = feedback['query']
        sql = feedback['sql']
        
        # Extract potential domain terms
        terms = self._extract_potential_terms(query)
        
        for term in terms:
            if term in self.domain_terms:
                self.domain_terms[term].usage_count += 1
                if feedback['is_correct']:
                    self.domain_terms[term].confidence = min(
                        1.0,
                        self.domain_terms[term].confidence + 0.1
                    )
            else:
                self.domain_terms[term] = DomainTerm(
                    term=term,
                    description=self._generate_term_description(term, query, sql),
                    related_entities=self._find_related_entities(term, sql),
                    usage_count=1,
                    confidence=0.5
                )

    def _update_query_templates(self, feedback: Dict):
        """Update query templates based on feedback"""
        if feedback['is_correct']:
            query = feedback['query']
            sql = feedback['sql']
            
            # Generate template
            template = self._generate_template(query, sql)
            
            # Update template if it doesn't exist or if this is a better version
            if template not in self.query_templates or self._is_better_template(
                template, self.query_templates[template], sql
            ):
                self.query_templates[template] = sql

    def _extract_parameters(self, query: str) -> List[str]:
        """Extract parameters from query"""
        # Find potential parameters (values that might change)
        tokens = query.split()
        parameters = []
        
        for token in tokens:
            if self._is_specific_value(token):
                parameters.append(token)
        
        return parameters

    def _generate_template(self, query: str, sql: str) -> str:
        """Generate a query template"""
        # Replace specific values with placeholders
        template = query
        for param in self._extract_parameters(query):
            template = template.replace(param, f"<{param}>")
        
        return template

    def _is_better_template(self, new_template: str, old_template: str, sql: str) -> bool:
        """Determine if new template is better than old one"""
        # Compare complexity, length, and success rate
        new_complexity = self._calculate_complexity(new_template)
        old_complexity = self._calculate_complexity(old_template)
        
        return new_complexity < old_complexity

    def _calculate_complexity(self, template: str) -> float:
        """Calculate template complexity"""
        # Consider factors like length, number of parameters, etc.
        length_factor = len(template) / 100
        param_factor = len(re.findall(r'<[^>]+>', template)) * 0.1
        
        return length_factor + param_factor

    def _is_specific_value(self, token: str) -> bool:
        """Determine if token is a specific value"""
        # Check if token is a number, date, or specific identifier
        return (
            token.isdigit() or
            re.match(r'\d{4}-\d{2}-\d{2}', token) or
            re.match(r'[A-Z][a-z]+', token)
        )

    def _extract_potential_terms(self, query: str) -> List[str]:
        """Extract potential domain terms from query"""
        # Use NLP to identify potential domain terms
        tokens = self.tokenizer.tokenize(query)
        potential_terms = []
        
        for token in tokens:
            if self._is_potential_term(token):
                potential_terms.append(token)
        
        return potential_terms

    def _is_potential_term(self, token: str) -> bool:
        """Determine if token is a potential domain term"""
        # Check if token is a noun or proper noun
        return (
            len(token) > 3 and
            token[0].isupper() and
            not token.isdigit()
        )

    def _generate_term_description(self, term: str, query: str, sql: str) -> str:
        """Generate description for a domain term"""
        # Use the classification model to generate a description
        context = f"{query} {sql}"
        inputs = self.tokenizer(context, return_tensors="pt", truncation=True)
        outputs = self.classification_model(**inputs)
        
        # Extract key phrases
        key_phrases = self._extract_key_phrases(context)
        
        return f"Domain term related to {' '.join(key_phrases[:3])}"

    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases from text"""
        # Use the embedding model to find important phrases
        sentences = text.split('.')
        embeddings = self.embedding_model.encode(sentences)
        
        # Use clustering to find important phrases
        clustering = DBSCAN(eps=0.3, min_samples=2).fit(embeddings)
        
        # Get the most representative phrases from each cluster
        key_phrases = []
        for label in set(clustering.labels_):
            if label != -1:  # Skip noise
                cluster_sentences = [s for s, l in zip(sentences, clustering.labels_) if l == label]
                key_phrases.extend(cluster_sentences)
        
        return key_phrases

    def _find_related_entities(self, term: str, sql: str) -> Set[str]:
        """Find entities related to a term in SQL"""
        # Extract table and column names from SQL
        entities = set()
        
        # Find tables
        table_matches = re.finditer(r'FROM\s+(\w+)', sql, re.IGNORECASE)
        entities.update(match.group(1) for match in table_matches)
        
        # Find columns
        column_matches = re.finditer(r'SELECT\s+(.*?)\s+FROM', sql, re.IGNORECASE)
        for match in column_matches:
            columns = match.group(1).split(',')
            entities.update(col.strip() for col in columns)
        
        return entities

    def get_similar_queries(self, query: str, limit: int = 5) -> List[Dict]:
        """Get similar queries from history"""
        query_embedding = self.embedding_model.encode(query)
        
        similar_queries = []
        for feedback in self.feedback_history:
            history_embedding = self.embedding_model.encode(feedback['query'])
            similarity = np.dot(query_embedding, history_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(history_embedding)
            )
            
            if similarity > 0.7:  # Similarity threshold
                similar_queries.append({
                    'query': feedback['query'],
                    'sql': feedback['sql'],
                    'similarity': similarity
                })
        
        # Sort by similarity and limit results
        return sorted(similar_queries, key=lambda x: x['similarity'], reverse=True)[:limit]

    def get_query_templates(self) -> Dict[str, str]:
        """Get available query templates"""
        return self.query_templates

    def get_domain_terms(self) -> Dict[str, DomainTerm]:
        """Get domain terminology"""
        return self.domain_terms

    def get_uncertain_queries(self) -> List[Dict]:
        """Get queries that need attention"""
        return self.uncertain_queries 