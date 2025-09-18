import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine
import os

def get_db_engine():
    """Conecta com o banco de dados usando variáveis de ambiente"""
    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")
    host = os.getenv("DB_HOST")
    db = os.getenv("DB_NAME")
    port = os.getenv("DB_PORT", 5432)
    url = f"postgresql://{user}:{password}@{host}:{port}/{db}"
    engine = create_engine(url)
    return engine

def read_table(table_name: str) -> pd.DataFrame:
    engine = get_db_engine()
    return pd.read_sql_table(table_name, engine)
