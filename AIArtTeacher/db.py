import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = os.getenv('AIAT_DATABASE_URL', 'sqlite:///./aiartteacher.db')

connect_args = {}
if DATABASE_URL.startswith('sqlite'):
    connect_args = {'check_same_thread': False}

engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def init_db():
    from .models import User, Submission, Feedback, Community, Challenge, OrientationPrompt, RefreshToken

    Base.metadata.create_all(bind=engine)
