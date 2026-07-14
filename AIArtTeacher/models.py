from datetime import datetime
from enum import Enum
from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey, JSON, Table
from sqlalchemy.orm import relationship
from .db import Base


class ExperienceLevel(str, Enum):
    beginner = 'beginner'
    intermediate = 'intermediate'
    advanced = 'advanced'


class UserRole(str, Enum):
    student = 'student'
    teacher = 'teacher'
    admin = 'admin'


class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    age = Column(Integer, nullable=False)
    experience_level = Column(String, nullable=False)
    role = Column(String, default=UserRole.student.value, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    submissions = relationship('Submission', back_populates='user')
    feedbacks = relationship('Feedback', back_populates='reviewer')


class Submission(Base):
    __tablename__ = 'submissions'

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    title = Column(String, nullable=False)
    description = Column(String, nullable=True)
    mode = Column(String, default='realistic', nullable=False)
    original_image_url = Column(String, nullable=False)
    reference_image_url = Column(String, nullable=True)
    uploaded_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    is_upside_down = Column(Boolean, default=False, nullable=False)
    submission_metadata = Column(JSON, default={}, nullable=False)
    deleted = Column(Boolean, default=False, nullable=False)

    user = relationship('User', back_populates='submissions')
    feedbacks = relationship('Feedback', back_populates='submission')
    challenge_entries = relationship('ChallengeEntry', back_populates='submission')

    @property
    def overall_score(self):
        latest = None
        if self.feedbacks:
            latest = sorted(self.feedbacks, key=lambda f: f.created_at, reverse=True)[0]
        return latest.overall_score if latest else None


class Feedback(Base):
    __tablename__ = 'feedbacks'

    id = Column(Integer, primary_key=True, index=True)
    submission_id = Column(Integer, ForeignKey('submissions.id'), nullable=False)
    reviewer_id = Column(Integer, ForeignKey('users.id'), nullable=True)
    scores = Column(JSON, default={}, nullable=False)
    overall_score = Column(Integer, nullable=True)
    suggestions = Column(String, nullable=True)
    comments = Column(JSON, default=[], nullable=False)
    annotated_image_url = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    submission = relationship('Submission', back_populates='feedbacks')
    reviewer = relationship('User', back_populates='feedbacks')


class Community(Base):
    __tablename__ = 'communities'

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    is_public = Column(Boolean, default=True, nullable=False)
    members = Column(JSON, default=[], nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class Challenge(Base):
    __tablename__ = 'challenges'

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    month = Column(String, nullable=False)
    start_at = Column(DateTime, nullable=False)
    end_at = Column(DateTime, nullable=False)
    leaderboard = Column(JSON, default=[], nullable=False)

    entries = relationship('ChallengeEntry', back_populates='challenge')


class ChallengeEntry(Base):
    __tablename__ = 'challenge_entries'

    id = Column(Integer, primary_key=True, index=True)
    challenge_id = Column(Integer, ForeignKey('challenges.id'), nullable=False)
    submission_id = Column(Integer, ForeignKey('submissions.id'), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    challenge = relationship('Challenge', back_populates='entries')
    submission = relationship('Submission', back_populates='challenge_entries')


class OrientationPrompt(Base):
    __tablename__ = 'orientation_prompts'

    id = Column(Integer, primary_key=True, index=True)
    submission_id = Column(Integer, ForeignKey('submissions.id'), nullable=False)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    detected_rotation_confidence = Column(Integer, nullable=False)
    prompt_text = Column(String, nullable=False)
    user_response = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class RefreshToken(Base):
    __tablename__ = 'refresh_tokens'

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    token = Column(String, nullable=False, unique=True)
    expires_at = Column(DateTime, nullable=False)
