import json
from typing import Dict, List, Optional
import oracledb
from dataclasses import dataclass
from collections import defaultdict
import re

@dataclass
class Column:
    name: str
    data_type: str
    nullable: bool
    description: Optional[str] = None

@dataclass
class Table:
    name: str
    columns: List[Column]
    primary_key: List[str]
    foreign_keys: Dict[str, str]  # column -> referenced table.column
    description: Optional[str] = None

class SchemaManager:
    def __init__(self, connection: oracledb.Connection, table_filter: Optional[List[str]] = None, use_specific_tables: bool = False):
        """Initialize SchemaManager with optional table filter
        
        Args:
            connection: Oracle database connection
            table_filter: Optional list of specific tables to include
            use_specific_tables: If True, will initialize with specific BP-related tables
        """
        self.connection = connection
        self.table_filter = table_filter
        self.schema_info = {}
        self.relationships = {}
        
        # Initialize schema based on parameters
        if use_specific_tables:
            self.initialize_with_specific_tables()
        else:
            self.initialize_schema()

    def initialize_schema(self):
        """Initialize schema information with optional table filtering"""
        try:
            # Get all tables (filtered if table_filter is provided)
            tables_query = """
            SELECT table_name 
            FROM user_tables 
            WHERE table_name NOT LIKE 'BIN$%'
            """
            if self.table_filter:
                tables_query += f" AND table_name IN ({','.join([':1'] * len(self.table_filter))})"
                cursor = self.connection.cursor()
                cursor.execute(tables_query, self.table_filter)
            else:
                cursor = self.connection.cursor()
                cursor.execute(tables_query)
            
            tables = [row[0] for row in cursor.fetchall()]
            
            # Get columns for filtered tables
            for table in tables:
                self.schema_info[table] = self._get_table_info(table)
            
            # Get relationships between filtered tables
            self._analyze_relationships()
            
            logger.info(f"Schema initialized with {len(tables)} tables")
            if self.table_filter:
                logger.info(f"Filtered tables: {', '.join(self.table_filter)}")
                
        except Exception as e:
            logger.error(f"Error initializing schema: {str(e)}")
            raise

    def add_tables(self, tables: List[str]):
        """Add additional tables to the schema"""
        try:
            # Add new tables to filter
            if self.table_filter is None:
                self.table_filter = []
            self.table_filter.extend(tables)
            
            # Reinitialize schema with new tables
            self.initialize_schema()
            
            logger.info(f"Added tables to schema: {', '.join(tables)}")
        except Exception as e:
            logger.error(f"Error adding tables: {str(e)}")
            raise

    def remove_tables(self, tables: List[str]):
        """Remove tables from the schema"""
        try:
            if self.table_filter:
                self.table_filter = [t for t in self.table_filter if t not in tables]
                # Remove from schema info
                for table in tables:
                    self.schema_info.pop(table, None)
                # Reanalyze relationships
                self._analyze_relationships()
                
                logger.info(f"Removed tables from schema: {', '.join(tables)}")
        except Exception as e:
            logger.error(f"Error removing tables: {str(e)}")
            raise

    def get_filtered_schema(self) -> Dict[str, Any]:
        """Get schema information for filtered tables"""
        return {
            'tables': self.schema_info,
            'relationships': self.relationships
        }

    def get_complete_schema(self) -> Dict[str, Table]:
        """Extract complete schema information including relationships"""
        cursor = self.connection.cursor()
        schema = {}

        # Get all tables
        cursor.execute("""
            SELECT table_name, comments 
            FROM user_tab_comments 
            WHERE table_type = 'TABLE'
        """)
        tables = cursor.fetchall()

        for table_name, description in tables:
            # Get columns
            cursor.execute("""
                SELECT column_name, data_type, nullable, comments
                FROM user_tab_columns
                WHERE table_name = :1
                ORDER BY column_id
            """, [table_name])
            columns = [
                Column(
                    name=col[0],
                    data_type=col[1],
                    nullable=col[2] == 'Y',
                    description=col[3]
                )
                for col in cursor.fetchall()
            ]

            # Get primary key
            cursor.execute("""
                SELECT cols.column_name
                FROM all_constraints cons, all_cons_columns cols
                WHERE cons.constraint_type = 'P'
                AND cons.constraint_name = cols.constraint_name
                AND cons.owner = cols.owner
                AND cons.table_name = :1
                ORDER BY cols.position
            """, [table_name])
            primary_key = [row[0] for row in cursor.fetchall()]

            # Get foreign keys
            cursor.execute("""
                SELECT a.column_name, c_pk.table_name || '.' || c_pk.column_name
                FROM all_cons_columns a
                JOIN all_constraints c ON a.owner = c.owner
                AND a.constraint_name = c.constraint_name
                JOIN all_constraints c_pk ON c.r_owner = c_pk.owner
                AND c.r_constraint_name = c_pk.constraint_name
                JOIN all_cons_columns c_pk_col ON c_pk.owner = c_pk_col.owner
                AND c_pk.constraint_name = c_pk_col.constraint_name
                AND c_pk_col.position = a.position
                WHERE c.constraint_type = 'R'
                AND a.table_name = :1
            """, [table_name])
            foreign_keys = {row[0]: row[1] for row in cursor.fetchall()}

            schema[table_name] = Table(
                name=table_name,
                columns=columns,
                primary_key=primary_key,
                foreign_keys=foreign_keys,
                description=description
            )

        return schema

    def get_schema_statistics(self) -> Dict[str, Dict]:
        """Get basic statistics for each table"""
        cursor = self.connection.cursor()
        stats = {}

        for table_name in self.get_complete_schema().keys():
            cursor.execute(f"""
                SELECT COUNT(*) as row_count,
                       COUNT(DISTINCT {', '.join(self.get_complete_schema()[table_name].primary_key)}) as unique_keys
                FROM {table_name}
            """)
            row_count, unique_keys = cursor.fetchone()
            
            stats[table_name] = {
                'row_count': row_count,
                'unique_keys': unique_keys
            }

        return stats

    def format_schema_for_llm(self, tables: Optional[List[str]] = None) -> str:
        """Format schema information for LLM consumption"""
        schema = self.get_complete_schema()
        formatted = []

        # If specific tables requested, filter schema
        if tables:
            schema = {k: v for k, v in schema.items() if k in tables}

        for table_name, table in schema.items():
            table_info = [f"Table: {table_name}"]
            if table.description:
                table_info.append(f"Description: {table.description}")
            
            table_info.append("Columns:")
            for col in table.columns:
                col_info = f"  - {col.name} ({col.data_type})"
                if not col.nullable:
                    col_info += " NOT NULL"
                if col.description:
                    col_info += f" -- {col.description}"
                table_info.append(col_info)

            if table.primary_key:
                table_info.append(f"Primary Key: {', '.join(table.primary_key)}")
            
            if table.foreign_keys:
                table_info.append("Foreign Keys:")
                for col, ref in table.foreign_keys.items():
                    table_info.append(f"  - {col} -> {ref}")

            formatted.append("\n".join(table_info))

        return "\n\n".join(formatted)

    def get_relevant_tables(self, query: str) -> List[str]:
        """Identify relevant tables for a given query using simple keyword matching"""
        # This is a simple implementation. In production, you'd want to use
        # more sophisticated methods like embeddings or ML models
        schema = self.get_complete_schema()
        relevant_tables = set()

        # Convert query to lowercase for case-insensitive matching
        query_lower = query.lower()

        for table_name, table in schema.items():
            # Check table name
            if table_name.lower() in query_lower:
                relevant_tables.add(table_name)
                continue

            # Check column names
            for column in table.columns:
                if column.name.lower() in query_lower:
                    relevant_tables.add(table_name)
                    break

            # Check table description
            if table.description and table.description.lower() in query_lower:
                relevant_tables.add(table_name)

        return list(relevant_tables)

    def get_schema_context(self, query: str) -> str:
        """Get relevant schema context for a query"""
        relevant_tables = self.get_relevant_tables(query)
        return self.format_schema_for_llm(relevant_tables)

    def save_schema_to_file(self, filepath: str):
        """Save schema information to a JSON file"""
        schema = self.get_complete_schema()
        stats = self.get_schema_statistics()

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
                    'foreign_keys': table.foreign_keys,
                    'statistics': stats.get(name, {})
                }
                for name, table in schema.items()
            }
        }

        with open(filepath, 'w') as f:
            json.dump(schema_data, f, indent=2)

    def get_tables_by_pattern(self, pattern: str) -> List[str]:
        """Get all tables matching a specific pattern"""
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                SELECT table_name 
                FROM user_tables 
                WHERE UPPER(table_name) LIKE UPPER(:1)
                AND table_name NOT LIKE 'BIN$%'
            """, [f'%{pattern}%'])
            return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error getting tables by pattern: {str(e)}")
            raise

    def initialize_with_specific_tables(self):
        """Initialize schema with specific tables and BP-related tables"""
        try:
            # Specific tables to include
            specific_tables = [
                'obj_bp',
                'obj_key',
                'obj_class',
                'obj_classif',
                'code_obj_class',
                'code_obj_classif'
            ]
            
            # Get all tables with 'bp' in their name
            bp_tables = self.get_tables_by_pattern('bp')
            
            # Combine and remove duplicates
            all_tables = list(set(specific_tables + bp_tables))
            
            # Initialize schema with these tables
            self.table_filter = all_tables
            self.initialize_schema()
            
            logger.info(f"Initialized schema with {len(all_tables)} tables:")
            logger.info(f"Specific tables: {', '.join(specific_tables)}")
            logger.info(f"BP-related tables: {', '.join(bp_tables)}")
            
        except Exception as e:
            logger.error(f"Error initializing with specific tables: {str(e)}")
            raise

    def clear_table_filter(self):
        """Clear the table filter and reinitialize schema with all tables"""
        try:
            self.table_filter = None
            self.initialize_schema()
            logger.info("Table filter cleared. Schema now includes all tables.")
        except Exception as e:
            logger.error(f"Error clearing table filter: {str(e)}")
            raise 