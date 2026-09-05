"""
SQLAlchemy ORM models and database session management.
Defines schema for survey missions, sonar image records, target detections, and tracks.
"""

import os
from datetime import datetime
from pathlib import Path
from sqlalchemy import (
    create_engine, Column, Integer, Float, String,
    Text, DateTime, Boolean, ForeignKey, JSON
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship, Session
from sqlalchemy.pool import StaticPool

Base = declarative_base()


class Mission(Base):
    __tablename__ = "missions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    mission_id = Column(String(64), unique=True, nullable=False)
    name = Column(String(256), nullable=False)
    date = Column(String(32))
    area = Column(String(256))
    status = Column(String(32), default="active")
    operator = Column(String(128))
    vehicle = Column(String(128))
    sonar_config = Column(String(256))
    data_label = Column(String(64), default="SURVEY")
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    images = relationship("SonarImage", back_populates="mission", cascade="all, delete-orphan")


class SonarImage(Base):
    __tablename__ = "sonar_images"

    id = Column(Integer, primary_key=True, autoincrement=True)
    image_id = Column(String(64), unique=True, nullable=False)
    mission_id = Column(String(64), ForeignKey("missions.mission_id"), nullable=False)
    filename = Column(String(512))
    filepath = Column(String(1024))
    scenario_type = Column(String(64))
    description = Column(Text)
    quality_score = Column(Float, default=0.0)
    quality_label = Column(String(16), default="UNKNOWN")
    width = Column(Integer)
    height = Column(Integer)
    lat = Column(Float)
    lon = Column(Float)
    depth_m = Column(Float)
    data_label = Column(String(64), default="SURVEY")
    coordinates_label = Column(String(64), default="GPS")
    preprocessed_path = Column(String(1024))
    result_path = Column(String(1024))
    processed = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    mission = relationship("Mission", back_populates="images")
    detections = relationship("Detection", back_populates="image", cascade="all, delete-orphan")


class Detection(Base):
    __tablename__ = "detections"

    id = Column(Integer, primary_key=True, autoincrement=True)
    detection_id = Column(String(64), unique=True, nullable=False)
    image_id = Column(String(64), ForeignKey("sonar_images.image_id"), nullable=False)
    mission_id = Column(String(64))

    # Target class & score
    class_name = Column(String(64))
    confidence = Column(Float, default=0.0)
    is_anomaly = Column(Boolean, default=False)
    anomaly_score = Column(Float, default=0.0)

    # Shadow metrics
    shadow_score = Column(Float, default=0.0)
    shadow_area = Column(Float, default=0.0)
    shadow_to_object_ratio = Column(Float, default=0.0)

    # Evidence fusion
    evidence_score = Column(Float, default=0.0)
    severity = Column(String(16), default="UNKNOWN")

    # Spatial bounding box
    bbox_x1 = Column(Integer)
    bbox_y1 = Column(Integer)
    bbox_x2 = Column(Integer)
    bbox_y2 = Column(Integer)
    object_area = Column(Integer, default=0)

    # Positional data
    lat = Column(Float)
    lon = Column(Float)
    depth_m = Column(Float)
    coordinates_label = Column(String(64), default="GPS")

    # Metadata
    mode = Column(String(32), default="DEMO")
    model_version = Column(String(64))
    track_id = Column(String(64))
    data_label = Column(String(64), default="SURVEY")

    # Review & verification
    operator_status = Column(String(32), default="pending")
    operator_label = Column(String(64))
    operator_note = Column(Text)
    reviewed_at = Column(DateTime)

    created_at = Column(DateTime, default=datetime.utcnow)

    image = relationship("SonarImage", back_populates="detections")


class Track(Base):
    __tablename__ = "tracks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    track_id = Column(String(64), unique=True, nullable=False)
    mission_id = Column(String(64))
    class_name = Column(String(64))
    frame_start = Column(Integer)
    frame_end = Column(Integer)
    num_frames = Column(Integer, default=1)
    status = Column(String(32), default="active")
    detection_ids = Column(JSON)
    lat = Column(Float)
    lon = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)


_engine = None
_SessionLocal = None


def get_engine(db_path: str = "sih26057.db"):
    global _engine
    if _engine is None:
        db_url = f"sqlite:///{db_path}"
        _engine = create_engine(
            db_url,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(_engine)
    return _engine


def get_session(db_path: str = "sih26057.db") -> Session:
    global _SessionLocal
    engine = get_engine(db_path)
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    return _SessionLocal()


def init_db(db_path: str = "sih26057.db"):
    """Initialize database and ensure tables exist."""
    engine = get_engine(db_path)
    Base.metadata.create_all(engine)
    return engine
