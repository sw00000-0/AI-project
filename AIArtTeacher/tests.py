import importlib
import os
from pathlib import Path

package_dir = Path(__file__).resolve().parent
test_db_path = package_dir / 'test_aiartteacher.db'
os.environ['AIAT_DATABASE_URL'] = f'sqlite:///{test_db_path}'

import pytest
from fastapi.testclient import TestClient

import AIArtTeacher.db as db_module
importlib.reload(db_module)

from AIArtTeacher.main import app
from AIArtTeacher.db import init_db, SessionLocal
from AIArtTeacher.models import User, Submission, Challenge
from AIArtTeacher.auth import get_password_hash

client = TestClient(app)

@pytest.fixture(scope='session', autouse=True)
def setup_db():
    for candidate in [Path.cwd() / 'test_aiartteacher.db', test_db_path]:
        if candidate.exists():
            candidate.unlink()
    init_db()
    db = SessionLocal()
    try:
        user = User(
            name='Test User',
            email='test@example.com',
            password_hash=get_password_hash('password123'),
            age=22,
            experience_level='advanced',
        )
        db.add(user)
        db.commit()
    finally:
        db.close()
    yield
    try:
        if test_db_path.exists():
            test_db_path.unlink()
    except FileNotFoundError:
        pass


def test_signup_and_login():
    response = client.post('/api/auth/signup', json={
        'name': 'Alice',
        'email': 'alice@example.com',
        'password': 'P@ssw0rd',
        'age': 20,
        'experience_level': 'advanced'
    })
    assert response.status_code == 201
    assert response.json()['experience_level'] == 'advanced'

    login_resp = client.post('/api/auth/login', data={'username': 'alice@example.com', 'password': 'P@ssw0rd'})
    assert login_resp.status_code == 200
    assert 'access_token' in login_resp.json()


def test_create_submission_and_fetch_with_feedback():
    login_resp = client.post('/api/auth/login', data={'username': 'test@example.com', 'password': 'password123'})
    assert login_resp.status_code == 200
    token = login_resp.json()['access_token']
    headers = {'Authorization': f'Bearer {token}'}

    submission_resp = client.post('/api/submissions', json={
        'title': 'Test Art',
        'description': 'A sample upload',
        'mode': 'realistic',
        'original_image_url': 'https://example.com/test.jpg',
        'metadata': {'orientation_confidence': 92, 'orientation': 'rotated', 'width': 1024, 'height': 768, 'format': 'jpeg'}
    }, headers=headers)
    assert submission_resp.status_code == 201
    submission = submission_resp.json()
    assert submission['is_upside_down'] is True

    fetch_resp = client.get(f"/api/submissions/{submission['id']}", headers=headers)
    assert fetch_resp.status_code == 200
    assert fetch_resp.json()['id'] == submission['id']


def test_create_community_and_challenge_entry():
    login_resp = client.post('/api/auth/login', data={'username': 'test@example.com', 'password': 'password123'})
    token = login_resp.json()['access_token']
    headers = {'Authorization': f'Bearer {token}'}
    from datetime import datetime

    db = SessionLocal()
    try:
        challenge = Challenge(
            name='June Challenge',
            month='2026-06',
            start_at=datetime(2026, 6, 1, 0, 0, 0),
            end_at=datetime(2026, 6, 30, 23, 59, 59),
        )
        db.add(challenge)
        db.commit()
        db.refresh(challenge)
    finally:
        db.close()

    community_resp = client.post('/api/communities', json={'name': 'Artists Lounge', 'is_public': True}, headers=headers)
    assert community_resp.status_code == 201
    assert community_resp.json()['name'] == 'Artists Lounge'

    submission_resp = client.post('/api/submissions', json={
        'title': 'Challenge Entry',
        'description': 'Challenge submission',
        'mode': 'realistic',
        'original_image_url': 'https://example.com/challenge.jpg',
        'metadata': {'orientation_confidence': 10, 'orientation': 'upright', 'width': 1200, 'height': 900, 'format': 'png'}
    }, headers=headers)
    assert submission_resp.status_code == 201
    submission = submission_resp.json()

    entry_resp = client.post(f"/api/challenges/{challenge.month}/entries", json={'submission_id': submission['id']}, headers=headers)
    assert entry_resp.status_code == 201
    assert entry_resp.json()['month'] == challenge.month
