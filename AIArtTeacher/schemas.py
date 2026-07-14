from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, EmailStr, Field


class TokenData(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = 'bearer'


class UserCreate(BaseModel):
    name: str
    email: EmailStr
    password: str
    age: int
    experience_level: str


class UserResponse(BaseModel):
    id: int
    name: str
    email: EmailStr
    age: int
    experience_level: str
    role: str
    created_at: datetime

    class Config:
        orm_mode = True


class OrientationPromptResponse(BaseModel):
    id: int
    submission_id: int
    user_id: int
    detected_rotation_confidence: int
    prompt_text: str
    user_response: Optional[str]
    created_at: datetime

    class Config:
        orm_mode = True


class SubmissionCreate(BaseModel):
    title: str
    description: Optional[str] = None
    mode: Optional[str] = 'realistic'
    original_image_url: str
    reference_image_url: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class SubmissionSummary(BaseModel):
    id: int
    title: str
    mode: str
    original_image_url: str
    overall_score: Optional[int]
    uploaded_at: datetime

    class Config:
        orm_mode = True


class FeedbackCreate(BaseModel):
    reviewer_id: Optional[int]
    scores: Optional[Dict[str, int]]
    suggestions: Optional[str]
    annotated_image_url: Optional[str]
    comments: Optional[List[Dict[str, Any]]]


class FeedbackResponse(BaseModel):
    id: int
    submission_id: int
    reviewer_id: Optional[int]
    scores: Dict[str, int]
    overall_score: Optional[int]
    suggestions: Optional[str]
    comments: List[Dict[str, Any]]
    annotated_image_url: Optional[str]
    created_at: datetime

    class Config:
        orm_mode = True


class SubmissionResponse(BaseModel):
    id: int
    user_id: int
    title: str
    description: Optional[str]
    mode: str
    original_image_url: str
    reference_image_url: Optional[str]
    uploaded_at: datetime
    is_upside_down: bool
    submission_metadata: Dict[str, Any]
    deleted: bool
    overall_score: Optional[int] = None
    feedbacks: List[FeedbackResponse] = []

    model_config = {
        'from_attributes': True,
    }


class CommunityCreate(BaseModel):
    name: str
    is_public: bool = True


class CommunityResponse(BaseModel):
    id: int
    name: str
    is_public: bool
    members: List[int]
    created_at: datetime

    class Config:
        orm_mode = True


class ChallengeEntryCreate(BaseModel):
    submission_id: int


class ChallengeResponse(BaseModel):
    id: int
    name: str
    month: str
    start_at: datetime
    end_at: datetime
    leaderboard: List[Dict[str, Any]]

    class Config:
        orm_mode = True
