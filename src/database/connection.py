"""
Database Connection & Session Factory
Agentic Construction Risk Intelligence Platform
"""

import os
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from .models import Base, SiteRecord

# Default to SQLite for zero-setup execution, configurable to PostgreSQL via env
DEFAULT_DB_PATH = Path("data/construction_risk.db").resolve()
DEFAULT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DEFAULT_DB_PATH}")

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    """Create all tables and seed default site record if not exists."""
    Base.metadata.create_all(bind=engine)
    with SessionLocal() as session:
        default_site = session.query(SiteRecord).first()
        if not default_site:
            site = SiteRecord(
                name="Main Construction Site Alpha",
                location="Zone 1 - High Elevation Structure"
            )
            session.add(site)
            session.commit()
            print(f"[Database] Initialized tables and seeded default site: {site.name}")

def get_db():
    """FastAPI database session dependency."""
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()
