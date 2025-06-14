from typing import Optional, Dict, Any
import os
from anthropic import Anthropic
from dotenv import load_dotenv
import time
from tenacity import retry, stop_after_attempt, wait_exponential
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Claude model configuration
CLAUDE_MODEL = 'claude-3-sonnet-20240229'

class LLMManager:
    def __init__(self):
        # Initialize Claude
        self.anthropic = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        self.claude_model = CLAUDE_MODEL
        
        # Initialize rate limiting
        self.last_claude_call = 0
        self.min_call_interval = 0.1  # 100ms between calls
        
        logger.info("Successfully initialized LLM Manager with Claude Sonnet")
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _rate_limited_call(self, func, *args, **kwargs):
        """Execute a function with rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_claude_call
        if time_since_last < self.min_call_interval:
            time.sleep(self.min_call_interval - time_since_last)
        self.last_claude_call = time.time()
        
        return func(*args, **kwargs)
    
    def generate_sql(self, prompt: str) -> str:
        """Generate SQL using Claude Sonnet"""
        try:
            enhanced_prompt = f"""You are an expert Oracle SQL developer. Your task is to generate a valid Oracle SQL query based on the following context and requirements.

Context:
{prompt}

Requirements:
1. Generate the SQL query with a detailed explanation
2. Use Oracle-specific syntax and features
3. Include appropriate table aliases for better readability
4. Use proper JOIN syntax (INNER JOIN, LEFT JOIN, etc.)
5. Add appropriate indexes hints if needed
6. Use proper date/time functions for Oracle
7. Handle NULL values appropriately
8. Use proper string handling functions
9. Include appropriate WHERE clauses for filtering
10. Use proper aggregation functions when needed
11. Use Common Table Expressions (CTEs) when appropriate for:
    - Complex subqueries
    - Recursive queries
    - Multiple aggregations
    - Data preparation steps
    - Improving query readability

Available Oracle Features:
1. Common Table Expressions (CTEs):
   - WITH clause for named subqueries
   - Recursive CTEs for hierarchical data
   - Multiple CTEs in a single query
   - Materialized CTEs for performance
   - CTEs with window functions

2. Window Functions:
   - ROW_NUMBER(), RANK(), DENSE_RANK()
   - LAG(), LEAD()
   - FIRST_VALUE(), LAST_VALUE()
   - NTH_VALUE()
   - NTILE()

3. Analytic Functions:
   - OVER() clause with PARTITION BY and ORDER BY
   - ROWS/RANGE BETWEEN for window frames
   - CUME_DIST(), PERCENT_RANK()
   - LISTAGG() with OVER()
   - RATIO_TO_REPORT()

4. Advanced Analytics:
   - MODEL clause for spreadsheet-like calculations
   - MATCH_RECOGNIZE for pattern matching
   - PIVOT/UNPIVOT for data transformation
   - Hierarchical queries with CONNECT BY
   - Regular expressions with REGEXP functions

Example CTE Usage:
1. Simple CTE:
   WITH sales_data AS (
     SELECT product_id, SUM(amount) as total_sales
     FROM sales
     GROUP BY product_id
   )
   SELECT p.name, s.total_sales
   FROM products p
   JOIN sales_data s ON p.id = s.product_id;

2. Multiple CTEs:
   WITH 
   monthly_sales AS (
     SELECT product_id, 
            TRUNC(sale_date, 'MM') as month,
            SUM(amount) as total
     FROM sales
     GROUP BY product_id, TRUNC(sale_date, 'MM')
   ),
   product_ranks AS (
     SELECT product_id,
            month,
            RANK() OVER (PARTITION BY month ORDER BY total DESC) as rank
     FROM monthly_sales
   )
   SELECT p.name, pr.month, pr.rank
   FROM products p
   JOIN product_ranks pr ON p.id = pr.product_id
   WHERE pr.rank <= 5;

3. Recursive CTE:
   WITH RECURSIVE employee_hierarchy AS (
     SELECT id, name, manager_id, 1 as level
     FROM employees
     WHERE manager_id IS NULL
     UNION ALL
     SELECT e.id, e.name, e.manager_id, eh.level + 1
     FROM employees e
     JOIN employee_hierarchy eh ON e.manager_id = eh.id
   )
   SELECT * FROM employee_hierarchy;

Return the response in the following JSON format:
{{
    "sql": "the generated SQL query",
    "explanation": {{
        "overview": "Brief overview of what the query does",
        "tables": ["List of tables used and their purpose"],
        "joins": ["Explanation of join conditions and their purpose"],
        "conditions": ["Explanation of WHERE clause conditions"],
        "aggregations": ["Explanation of any aggregations or groupings"],
        "ctes": ["Explanation of any CTEs used and their purpose"],
        "window_functions": ["Explanation of any window functions used"],
        "performance_notes": ["Any performance considerations or tips"]
    }}
}}"""

            response = self._rate_limited_call(
                self.anthropic.messages.create,
                model=self.claude_model,
                max_tokens=2000,  # Increased token limit for explanations
                messages=[{"role": "user", "content": enhanced_prompt}]
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Error generating SQL with Claude: {str(e)}")
            raise
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze natural language query for intent and entities using Claude Sonnet"""
        try:
            enhanced_prompt = f"""You are an expert in analyzing database queries. Analyze the following query and provide a detailed breakdown in JSON format.

Query: {query}

Provide a JSON object with the following structure:
{{
    "intent": {{
        "type": "SELECT|INSERT|UPDATE|DELETE",
        "description": "Detailed description of the query intent"
    }},
    "entities": {{
        "tables": ["list of tables involved"],
        "columns": ["list of columns involved"],
        "relationships": ["list of table relationships"]
    }},
    "conditions": {{
        "filters": ["list of filtering conditions"],
        "joins": ["list of join conditions"],
        "grouping": ["list of grouping conditions"]
    }},
    "aggregations": {{
        "functions": ["list of aggregation functions"],
        "group_by": ["list of group by columns"],
        "having": ["list of having conditions"]
    }},
    "window_functions": {{
        "functions": ["list of window functions used"],
        "partition_by": ["list of partition columns"],
        "order_by": ["list of order by columns"],
        "window_frame": ["window frame specification if any"]
    }},
    "analytic_functions": {{
        "functions": ["list of analytic functions used"],
        "over_clause": ["details of OVER clause"],
        "window_specification": ["window specification details"]
    }},
    "advanced_features": {{
        "model_clause": ["details of MODEL clause if used"],
        "match_recognize": ["details of pattern matching if used"],
        "pivot_unpivot": ["details of PIVOT/UNPIVOT if used"],
        "hierarchical": ["details of hierarchical queries if used"],
        "regexp": ["details of regular expressions if used"]
    }},
    "complexity": {{
        "level": "SIMPLE|MODERATE|COMPLEX",
        "estimated_cost": "LOW|MEDIUM|HIGH",
        "optimization_needed": true|false,
        "performance_considerations": ["list of performance considerations"]
    }}
}}

Return ONLY the JSON object, nothing else."""

            response = self.generate_sql(enhanced_prompt)
            return eval(response)  # Convert string response to dict
        except Exception as e:
            logger.error(f"Error analyzing query: {str(e)}")
            raise
    
    def optimize_sql(self, sql: str) -> Dict[str, Any]:
        """Get SQL optimization suggestions using Claude Sonnet"""
        try:
            enhanced_prompt = f"""You are an expert in Oracle SQL optimization. Analyze the following SQL query and provide detailed optimization suggestions in JSON format.

SQL Query:
{sql}

Provide a JSON object with the following structure:
{{
    "index_recommendations": [
        {{
            "table": "table_name",
            "columns": ["column1", "column2"],
            "type": "B-tree|Bitmap|Function-based",
            "reason": "explanation"
        }}
    ],
    "query_structure": [
        {{
            "issue": "description of structural issue",
            "suggestion": "optimization suggestion",
            "impact": "HIGH|MEDIUM|LOW"
        }}
    ],
    "performance_considerations": [
        {{
            "aspect": "description of performance aspect",
            "recommendation": "specific recommendation",
            "expected_improvement": "estimated improvement"
        }}
    ],
    "best_practices": [
        {{
            "practice": "description of best practice",
            "current_usage": "how it's currently used",
            "recommended_usage": "how it should be used"
        }}
    ],
    "estimated_improvement": {{
        "execution_time": "estimated reduction in execution time",
        "resource_usage": "estimated reduction in resource usage",
        "scalability": "impact on query scalability"
    }}
}}

Return ONLY the JSON object, nothing else."""

            response = self.generate_sql(enhanced_prompt)
            return eval(response)  # Convert string response to dict
        except Exception as e:
            logger.error(f"Error optimizing SQL: {str(e)}")
            raise 