import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from sqlalchemy import Column, DateTime, Float, Integer, JSON, String, create_engine, select
from sqlalchemy.orm import Session, declarative_base

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
Base = declarative_base()


class PredictionRecord(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    endpoint = Column(String, nullable=False)
    user_id = Column(Integer, nullable=True)
    input_json = Column(JSON, nullable=False)
    prediction = Column(Float, nullable=False)
    lower_bound = Column(Float, nullable=False)
    upper_bound = Column(Float, nullable=False)
    model_version = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)


engine = create_engine(DATABASE_URL, future=True) if DATABASE_URL else None

if engine is not None:
    Base.metadata.create_all(bind=engine)


def log_prediction(
    *,
    endpoint: str,
    input_json: Dict[str, Any],
    prediction: float,
    lower_bound: float,
    upper_bound: float,
    model_version: str,
    user_id: Optional[int] = None,
) -> Optional[int]:
    if engine is None:
        raise RuntimeError("DATABASE_URL is not configured. Add it to your environment or .env file.")

    with Session(engine) as session:
        row = PredictionRecord(
            timestamp=datetime.utcnow(),
            endpoint=endpoint,
            user_id=user_id,
            input_json=input_json,
            prediction=float(prediction),
            lower_bound=float(lower_bound),
            upper_bound=float(upper_bound),
            model_version=model_version,
            created_at=datetime.utcnow(),
        )
        session.add(row)
        session.commit()
        session.refresh(row)
        return row.id


def get_recent_predictions(limit: int = 50) -> List[Dict[str, Any]]:
    if engine is None:
        return []

    with Session(engine) as session:
        rows = session.execute(
            select(PredictionRecord)
            .order_by(PredictionRecord.created_at.desc())
            .limit(limit)
        ).scalars().all()

    return [
        {
            "id": row.id,
            "timestamp": row.timestamp.isoformat() if row.timestamp else None,
            "endpoint": row.endpoint,
            "user_id": row.user_id,
            "input_json": row.input_json,
            "prediction": row.prediction,
            "lower_bound": row.lower_bound,
            "upper_bound": row.upper_bound,
            "model_version": row.model_version,
            "created_at": row.created_at.isoformat() if row.created_at else None,
        }
        for row in rows
    ]
