from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sql_generator import TextToSQLGenerator
from typing import Optional, List, Dict
import uvicorn

app = FastAPI(title="Text to SQL Generator API")
sql_generator = TextToSQLGenerator()

class QueryRequest(BaseModel):
    query: str
    use_advanced_features: bool = False
    hints: Optional[List[str]] = None

class QueryResponse(BaseModel):
    generated_sql: str
    is_valid: bool
    results: Optional[List] = None
    relevant_tables: List[str]
    optimizations: Optional[Dict] = None

class FeedbackRequest(BaseModel):
    query: str
    sql: str
    is_correct: bool
    corrected_sql: Optional[str] = None

class OptimizationRequest(BaseModel):
    sql_query: str

class SchemaResponse(BaseModel):
    tables: dict
    statistics: dict

class StoredProcedureResponse(BaseModel):
    procedures: List[Dict]
    package_procedures: List[Dict]

class DomainTermResponse(BaseModel):
    description: str
    related_entities: List[str]
    usage_count: int
    confidence: float

class LearningResponse(BaseModel):
    uncertain_queries: List[Dict]
    query_templates: Dict[str, str]
    domain_terms: Dict[str, DomainTermResponse]

@app.post("/generate", response_model=QueryResponse)
async def generate_sql(request: QueryRequest):
    try:
        # Generate SQL
        sql = sql_generator.generate_sql(request.query)
        
        # Add Oracle hints if requested
        if request.hints:
            sql = sql_generator.add_oracle_hints(sql, request.hints)
        
        # Validate SQL
        is_valid = sql_generator.validate_sql(sql)
        
        # Get relevant tables
        relevant_tables = sql_generator.schema_manager.get_relevant_tables(request.query)
        
        # Get optimizations if requested
        optimizations = None
        if request.use_advanced_features:
            optimizations = sql_generator.optimize_query(sql)
        
        # Execute query if valid
        results = None
        if is_valid:
            results = sql_generator.execute_query(sql)
        
        return QueryResponse(
            generated_sql=sql,
            is_valid=is_valid,
            results=results,
            relevant_tables=relevant_tables,
            optimizations=optimizations
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
async def add_feedback(request: FeedbackRequest):
    try:
        sql_generator.add_feedback(
            request.query,
            request.sql,
            request.is_correct,
            request.corrected_sql
        )
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/optimize", response_model=Dict)
async def optimize_query(request: OptimizationRequest):
    try:
        return sql_generator.optimize_query(request.sql_query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stored-procedures", response_model=StoredProcedureResponse)
async def get_stored_procedures():
    try:
        return sql_generator.get_stored_procedures()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/schema", response_model=SchemaResponse)
async def get_schema():
    try:
        schema = sql_generator.schema_manager.get_complete_schema()
        stats = sql_generator.schema_manager.get_schema_statistics()
        
        # Convert schema to dictionary format
        schema_dict = {
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
            for name, table in schema.items()
        }
        
        return SchemaResponse(
            tables=schema_dict,
            statistics=stats
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/schema/export")
async def export_schema():
    try:
        filepath = "schema_export.json"
        sql_generator.schema_manager.save_schema_to_file(filepath)
        return {"message": f"Schema exported to {filepath}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/learning", response_model=LearningResponse)
async def get_learning_data():
    """Get learning data including uncertain queries, templates, and domain terms"""
    try:
        uncertain_queries = sql_generator.get_uncertain_queries()
        query_templates = sql_generator.get_query_templates()
        domain_terms = sql_generator.get_domain_terms()
        
        return LearningResponse(
            uncertain_queries=uncertain_queries,
            query_templates=query_templates,
            domain_terms=domain_terms
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/learning/uncertain-queries")
async def get_uncertain_queries():
    """Get queries that need attention"""
    try:
        return sql_generator.get_uncertain_queries()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/learning/templates")
async def get_query_templates():
    """Get available query templates"""
    try:
        return sql_generator.get_query_templates()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/learning/domain-terms")
async def get_domain_terms():
    """Get domain terminology"""
    try:
        return sql_generator.get_domain_terms()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 