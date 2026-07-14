import os
from datetime import datetime
from pathlib import Path
from typing import List
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from .auth import get_db, authenticate_user, create_access_token, create_refresh_token, create_user, get_current_user
from .models import User, Submission, Feedback, Community, Challenge, ChallengeEntry, OrientationPrompt
from .schemas import UserCreate, UserResponse, TokenData, SubmissionCreate, SubmissionResponse, FeedbackCreate, FeedbackResponse, CommunityCreate, CommunityResponse, ChallengeEntryCreate, ChallengeResponse, OrientationPromptResponse
from .llm import generate_realistic_feedback, generate_creative_feedback
from .utils import generate_signed_upload_url, compute_overall_score

BASE_DIR = Path(__file__).resolve().parent
app = FastAPI()

app.mount('/static', StaticFiles(directory=str(BASE_DIR / 'static')), name='static')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


@app.get('/', response_class=HTMLResponse)
def home():
    index_path = BASE_DIR / 'templates' / 'index.html'
    return index_path.read_text(encoding='utf-8')


@app.post('/api/auth/signup', response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def signup(user_create: UserCreate, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.email == user_create.email).first()
    if existing:
        raise HTTPException(status_code=400, detail='Email already registered')
    user = create_user(db, user_create)
    return user


@app.post('/api/auth/login', response_model=TokenData)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=400, detail='Incorrect email or password')
    access_token = create_access_token({'sub': user.id})
    refresh_token, expires_at = create_refresh_token(user.id)
    return {
        'access_token': access_token,
        'refresh_token': refresh_token,
        'token_type': 'bearer',
    }


@app.get('/api/auth/me', response_model=UserResponse)
def me(user: User = Depends(get_current_user)):
    return user


@app.get('/api/uploads/signed-url')
def signed_upload(filename: str, content_type: str):
    if not filename or not content_type:
        raise HTTPException(status_code=400, detail='filename and content_type are required')
    return generate_signed_upload_url(filename, content_type)


@app.post('/api/submissions', response_model=SubmissionResponse, status_code=status.HTTP_201_CREATED)
def create_submission(payload: SubmissionCreate, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    submission = Submission(
        user_id=user.id,
        title=payload.title,
        description=payload.description,
        mode=payload.mode or 'realistic',
        original_image_url=payload.original_image_url,
        reference_image_url=payload.reference_image_url,
        submission_metadata=payload.metadata or {},
    )
    db.add(submission)
    db.commit()
    db.refresh(submission)

    confidence = payload.metadata.get('orientation_confidence', 0)
    if confidence > 90:
        prompt = OrientationPrompt(
            submission_id=submission.id,
            user_id=user.id,
            detected_rotation_confidence=int(confidence),
            prompt_text='Detected rotated image — keep orientation as-is or auto-rotate?',
        )
        submission.is_upside_down = True
        db.add(prompt)
        db.commit()
        db.refresh(submission)
    
    if submission.mode == 'creative':
        llm = generate_creative_feedback(user.experience_level, submission.submission_metadata, submission.reference_image_url)
        scores = llm['scores']
        feedback = Feedback(
            submission_id=submission.id,
            reviewer_id=None,
            scores=scores,
            overall_score=None,
            suggestions=llm['suggestions'],
        )
    else:
        llm = generate_realistic_feedback(user.experience_level, submission.submission_metadata, submission.reference_image_url)
        scores = llm['scores']
        feedback = Feedback(
            submission_id=submission.id,
            reviewer_id=None,
            scores=scores,
            overall_score=compute_overall_score(scores),
            suggestions=llm['suggestions'],
        )
    db.add(feedback)
    db.commit()
    db.refresh(feedback)
    return submission


@app.get('/api/submissions', response_model=List[SubmissionResponse])
def list_submissions(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    submissions = db.query(Submission).filter(Submission.user_id == user.id, Submission.deleted == False).all()
    return submissions


@app.get('/api/submissions/{submission_id}', response_model=SubmissionResponse)
def get_submission(submission_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    submission = db.query(Submission).filter(Submission.id == submission_id, Submission.deleted == False).first()
    if not submission:
        raise HTTPException(status_code=404, detail='Submission not found')
    return submission


@app.delete('/api/submissions/{submission_id}', status_code=status.HTTP_204_NO_CONTENT)
def delete_submission(submission_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    submission = db.query(Submission).filter(Submission.id == submission_id, Submission.user_id == user.id, Submission.deleted == False).first()
    if not submission:
        raise HTTPException(status_code=404, detail='Submission not found')
    submission.deleted = True
    db.add(submission)
    db.commit()
    return None


@app.post('/api/submissions/{submission_id}/feedback', response_model=FeedbackResponse, status_code=status.HTTP_201_CREATED)
def post_feedback(submission_id: int, payload: FeedbackCreate, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    submission = db.query(Submission).filter(Submission.id == submission_id, Submission.deleted == False).first()
    if not submission:
        raise HTTPException(status_code=404, detail='Submission not found')
    overall_score = None
    if payload.scores and 'color_contrast' in payload.scores:
        overall_score = compute_overall_score(payload.scores)
    feedback = Feedback(
        submission_id=submission.id,
        reviewer_id=payload.reviewer_id,
        scores=payload.scores or {},
        overall_score=overall_score,
        suggestions=payload.suggestions,
        comments=payload.comments or [],
        annotated_image_url=payload.annotated_image_url,
    )
    db.add(feedback)
    db.commit()
    db.refresh(feedback)
    return feedback


@app.post('/api/communities', response_model=CommunityResponse, status_code=status.HTTP_201_CREATED)
def create_community(payload: CommunityCreate, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    community = Community(
        name=payload.name,
        is_public=payload.is_public,
        members=[user.id],
    )
    db.add(community)
    db.commit()
    db.refresh(community)
    return community


@app.post('/api/challenges/{month}/entries', response_model=ChallengeResponse, status_code=status.HTTP_201_CREATED)
def create_challenge_entry(month: str, payload: ChallengeEntryCreate, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    challenge = db.query(Challenge).filter(Challenge.month == month).first()
    if not challenge:
        raise HTTPException(status_code=404, detail='Challenge not found')
    submission = db.query(Submission).filter(Submission.id == payload.submission_id, Submission.user_id == user.id, Submission.deleted == False).first()
    if not submission:
        raise HTTPException(status_code=404, detail='Submission not found')
    existing = db.query(ChallengeEntry).filter(ChallengeEntry.challenge_id == challenge.id, ChallengeEntry.submission_id == submission.id).first()
    if existing:
        raise HTTPException(status_code=400, detail='Submission already entered')
    entry = ChallengeEntry(challenge_id=challenge.id, submission_id=submission.id)
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return challenge


@app.post('/api/orientation-prompts/{prompt_id}/response', response_model=OrientationPromptResponse)
def respond_orientation_prompt(prompt_id: int, note: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    prompt = db.query(OrientationPrompt).filter(OrientationPrompt.id == prompt_id, OrientationPrompt.user_id == user.id).first()
    if not prompt:
        raise HTTPException(status_code=404, detail='Prompt not found')
    prompt.user_response = note
    db.add(prompt)
    db.commit()
    db.refresh(prompt)
    return prompt
