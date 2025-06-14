# Oracle Text-to-SQL Generator

An AI-powered tool that converts natural language queries into Oracle SQL statements, with advanced features for schema understanding and query optimization.

## Features

- Natural language to SQL conversion using Claude Sonnet
- Intelligent schema understanding and relationship mapping
- Support for Oracle-specific features:
  - Window functions
  - Analytic functions
  - Advanced analytics (MODEL clause, pattern matching)
- Query optimization and performance suggestions
- Detailed query explanations
- Flexible table filtering and schema management

## Prerequisites

- Python 3.8+
- Oracle Database
- Anthropic API key
- Oracle Client libraries

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/oracle-text-to-sql.git
cd oracle-text-to-sql
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file with your configuration:
```env
# Oracle Database
ORACLE_USER=your_username
ORACLE_PASSWORD=your_password
ORACLE_HOST=your_host
ORACLE_PORT=1521
ORACLE_SERVICE=your_service

# Anthropic
ANTHROPIC_API_KEY=your_api_key

# Model Configuration
CLAUDE_MODEL=claude-3-sonnet-20240229
```

## Usage

1. Initialize the TextToSQLGenerator:
```python
from sql_generator import TextToSQLGenerator
from schema_manager import SchemaManager
import oracledb

# Create database connection
connection = oracledb.connect(
    user=os.getenv('ORACLE_USER'),
    password=os.getenv('ORACLE_PASSWORD'),
    host=os.getenv('ORACLE_HOST'),
    port=os.getenv('ORACLE_PORT'),
    service_name=os.getenv('ORACLE_SERVICE')
)

# Initialize with specific tables
schema_manager = SchemaManager(connection, use_specific_tables=True)

# Create generator
generator = TextToSQLGenerator(connection)
```

2. Generate SQL from natural language:
```python
query = "Show me the top 5 employees by salary in each department"
result = generator.generate_sql(query)
print(result['sql'])
print(result['explanation'])
```

## Project Structure

- `sql_generator.py`: Main SQL generation logic
- `schema_manager.py`: Database schema management
- `llm_manager.py`: LLM integration and query analysis
- `api.py`: FastAPI endpoints
- `main.py`: Application entry point

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Anthropic for Claude API
- Oracle for database features
- FastAPI for the web framework 