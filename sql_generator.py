import os
from typing import Optional, Dict, List
import oracledb
from dotenv import load_dotenv
import json
from datetime import datetime
from schema_manager import SchemaManager
from schema_ai import SchemaAI
from learning_manager import LearningManager
from llm_manager import LLMManager
import numpy as np

class TextToSQLGenerator:
    def __init__(self):
        load_dotenv()
        self.setup_database_connection()
        self.schema_manager = SchemaManager(self.connection)
        self.schema_ai = SchemaAI()
        self.learning_manager = LearningManager()
        self.llm_manager = LLMManager()  # Initialize LLM Manager
        self.query_cache = {}
        self.feedback_log = []
        
        # Initialize AI understanding of schema
        self.initialize_schema_understanding()
        
    def initialize_schema_understanding(self):
        """Initialize AI understanding of the schema"""
        # Get complete schema
        schema_data = {
            'tables': {
                name: {
                    'description': table.description,
                    'columns': [
                        {
                            'name': col.name,
                            'data_type': col.data_type,
                            'nullable': col.nullable,
                            'description': col.description
                        }
                        for col in table.columns
                    ],
                    'primary_key': table.primary_key,
                    'foreign_keys': table.foreign_keys
                }
                for name, table in self.schema_manager.get_complete_schema().items()
            }
        }
        
        # Analyze schema with AI
        self.schema_ai.analyze_schema(schema_data)
        self.schema_ai.cluster_entities()
        
        # Save analysis for future use
        self.schema_ai.save_analysis('schema_analysis.json')

    def setup_database_connection(self):
        """Initialize Oracle database connection"""
        try:
            # Initialize Oracle client
            oracledb.init_oracle_client()
            
            # Create connection
            self.connection = oracledb.connect(
                user=os.getenv('ORACLE_USER'),
                password=os.getenv('ORACLE_PASSWORD'),
                dsn=f"{os.getenv('ORACLE_HOST')}:{os.getenv('ORACLE_PORT')}/{os.getenv('ORACLE_SERVICE')}",
                encoding="UTF-8"
            )
            print("Successfully connected to Oracle Database")
        except Exception as e:
            print(f"Error connecting to Oracle Database: {str(e)}")
            raise

    def create_prompt(self, text_query: str) -> str:
        """Create a prompt for the LLM with schema context and examples"""
        # Understand query using AI
        query_understanding = self.schema_ai.understand_query(text_query)
        
        # Get relevant tables from AI understanding
        relevant_tables = set()
        for entity in query_understanding['matched_entities']:
            if self.schema_ai.entities[entity].type == 'table':
                relevant_tables.add(entity)
            else:
                # If it's a column, add its parent table
                relevant_tables.update(self.schema_ai.entities[entity].related_entities)
        
        # Get schema context
        schema_context = self.schema_manager.format_schema_for_llm(list(relevant_tables))
        
        # Add AI understanding context
        ai_context = f"""Query Understanding:
- Key Terms: {', '.join(query_understanding['query_terms'])}
- Related Entities: {', '.join(query_understanding['matched_entities'])}
"""
        
        # Get similar queries from learning manager
        similar_queries = self.learning_manager.get_similar_queries(text_query)
        examples_text = "\n\nSimilar Queries:\n" + "\n\n".join(
            f"Query: {q['query']}\nSQL: {q['sql']}\nSimilarity: {q['similarity']:.2f}"
            for q in similar_queries
        ) if similar_queries else ""
        
        # Get relevant domain terms
        domain_terms = self.learning_manager.get_domain_terms()
        relevant_terms = {
            term: data for term, data in domain_terms.items()
            if term.lower() in text_query.lower()
        }
        
        domain_context = "\n\nDomain Terms:\n" + "\n".join(
            f"- {term}: {data.description} (Confidence: {data.confidence:.2f})"
            for term, data in relevant_terms.items()
        ) if relevant_terms else ""
        
        # Create the prompt
        prompt = f"""Oracle SQL generator. Given the following schema and query understanding:

Schema:
{schema_context}

{ai_context}

{domain_context}

{examples_text}

Convert this natural language query to Oracle SQL:
{text_query}

Return ONLY the SQL, no explanations."""
        
        return prompt

    def generate_sql(self, text_query: str) -> str:
        """Generate SQL query from natural language text"""
        # Check cache first
        if text_query in self.query_cache:
            return self.query_cache[text_query]
        
        # Create prompt
        prompt = self.create_prompt(text_query)
        
        # Generate SQL using Claude Sonnet
        generated_sql = self.llm_manager.generate_sql(prompt)
        
        # Cache the result
        self.query_cache[text_query] = generated_sql
        
        return generated_sql

    def validate_sql(self, sql_query: str) -> bool:
        """Validate if the generated SQL query is valid"""
        try:
            # Check for DDL statements
            if any(keyword in sql_query.upper() for keyword in ['CREATE', 'ALTER', 'DROP', 'TRUNCATE']):
                print("DDL statements are not allowed")
                return False
                
            # Check for unauthorized tables
            cursor = self.connection.cursor()
            cursor.execute("""
                SELECT table_name 
                FROM user_tables
            """)
            authorized_tables = {row[0].upper() for row in cursor.fetchall()}
            
            # Simple table name extraction (in production, use proper SQL parsing)
            words = sql_query.upper().split()
            for i, word in enumerate(words):
                if word == 'FROM' and i + 1 < len(words):
                    table = words[i + 1].strip(';')
                    if table not in authorized_tables:
                        print(f"Unauthorized table: {table}")
                        return False
            
            # Validate query syntax
            cursor.execute(f"EXPLAIN PLAN FOR {sql_query}")
            return True
        except Exception as e:
            print(f"Invalid SQL query: {str(e)}")
            return False

    def execute_query(self, sql_query: str) -> Optional[list]:
        """Execute the SQL query and return results"""
        try:
            cursor = self.connection.cursor()
            cursor.execute(sql_query)
            results = cursor.fetchall()
            return results
        except Exception as e:
            print(f"Error executing query: {str(e)}")
            return None

    def add_feedback(self, query: str, sql: str, is_correct: bool, corrected_sql: Optional[str] = None):
        """Add feedback for query generation"""
        feedback = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'sql': sql,
            'is_correct': is_correct,
            'corrected_sql': corrected_sql
        }
        
        # Add to feedback log
        self.feedback_log.append(feedback)
        self.save_feedback_log()
        
        # Analyze feedback for learning
        self.learning_manager.analyze_feedback(feedback)

    def save_feedback_log(self):
        """Save feedback log to file"""
        with open('feedback_log.json', 'w') as f:
            json.dump(self.feedback_log, f, indent=2)

    def load_feedback_log(self):
        """Load feedback log from file"""
        try:
            with open('feedback_log.json', 'r') as f:
                self.feedback_log = json.load(f)
        except FileNotFoundError:
            self.feedback_log = []

    def optimize_query(self, sql_query: str) -> Dict[str, str]:
        """Analyze and suggest optimizations for the SQL query"""
        try:
            cursor = self.connection.cursor()
            
            # Get execution plan
            cursor.execute(f"EXPLAIN PLAN FOR {sql_query}")
            cursor.execute("""
                SELECT * FROM TABLE(DBMS_XPLAN.DISPLAY)
            """)
            execution_plan = cursor.fetchall()
            
            # Get table statistics
            cursor.execute("""
                SELECT table_name, num_rows, blocks, avg_row_len
                FROM user_tables
                WHERE table_name IN (
                    SELECT table_name 
                    FROM user_tab_columns 
                    WHERE column_name IN (
                        SELECT column_name 
                        FROM user_tab_columns 
                        WHERE table_name IN (
                            SELECT table_name 
                            FROM user_tables
                        )
                    )
                )
            """)
            table_stats = cursor.fetchall()
            
            # Analyze for potential optimizations
            optimizations = {
                'suggestions': [],
                'execution_plan': execution_plan,
                'table_statistics': table_stats
            }
            
            # Check for missing indexes
            if 'TABLE ACCESS FULL' in str(execution_plan):
                optimizations['suggestions'].append(
                    "Consider adding indexes for frequently queried columns"
                )
            
            # Check for parallel execution opportunities
            if len(table_stats) > 1 and any(stat[1] > 1000000 for stat in table_stats):
                optimizations['suggestions'].append(
                    "Consider using parallel execution for large tables"
                )
            
            # Check for materialized view opportunities
            if 'GROUP BY' in sql_query.upper() or 'JOIN' in sql_query.upper():
                optimizations['suggestions'].append(
                    "Consider creating a materialized view for frequently executed aggregations"
                )
            
            return optimizations
            
        except Exception as e:
            print(f"Error optimizing query: {str(e)}")
            return {'error': str(e)}

    def get_stored_procedures(self) -> List[Dict]:
        """Get list of available stored procedures and functions"""
        try:
            cursor = self.connection.cursor()
            
            # Get procedures
            cursor.execute("""
                SELECT object_name, procedure_name, arguments
                FROM user_procedures
                WHERE object_type IN ('PROCEDURE', 'FUNCTION')
                ORDER BY object_name, procedure_name
            """)
            procedures = cursor.fetchall()
            
            # Get package procedures
            cursor.execute("""
                SELECT package_name, object_name, procedure_name, arguments
                FROM user_procedures
                WHERE object_type = 'PACKAGE'
                ORDER BY package_name, object_name
            """)
            package_procedures = cursor.fetchall()
            
            return {
                'procedures': [
                    {
                        'name': p[0],
                        'type': 'PROCEDURE',
                        'arguments': p[2]
                    }
                    for p in procedures
                ],
                'package_procedures': [
                    {
                        'package': p[0],
                        'name': p[1],
                        'type': 'PACKAGE',
                        'arguments': p[3]
                    }
                    for p in package_procedures
                ]
            }
            
        except Exception as e:
            print(f"Error getting stored procedures: {str(e)}")
            return {'error': str(e)}

    def add_oracle_hints(self, sql_query: str, hints: List[str]) -> str:
        """Add Oracle hints to the SQL query"""
        # Find the first SELECT statement
        select_pos = sql_query.upper().find('SELECT')
        if select_pos == -1:
            return sql_query
            
        # Add hints after SELECT
        hints_str = ' /*+ ' + ' '.join(hints) + ' */'
        return sql_query[:select_pos + 6] + hints_str + sql_query[select_pos + 6:]

    def get_uncertain_queries(self) -> List[Dict]:
        """Get queries that need attention"""
        return self.learning_manager.get_uncertain_queries()

    def get_query_templates(self) -> Dict[str, str]:
        """Get available query templates"""
        return self.learning_manager.get_query_templates()

    def get_domain_terms(self) -> Dict[str, Dict]:
        """Get domain terminology"""
        return {
            term: {
                'description': data.description,
                'related_entities': list(data.related_entities),
                'usage_count': data.usage_count,
                'confidence': data.confidence
            }
            for term, data in self.learning_manager.get_domain_terms().items()
        }

    def __del__(self):
        """Cleanup database connection"""
        if hasattr(self, 'connection'):
            self.connection.close() 