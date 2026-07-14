import os
from datetime import datetime, timedelta
from passlib.context import CryptContext
from jose import JWTError, jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from .db import SessionLocal
from . import models

SECRET_KEY = os.getenv('AIAT_SECRET_KEY', 'change_this_secret')
ALGORITHM = 'HS256'
ACCESS_TOKEN_EXPIRE_MINUTES = 60
REFRESH_TOKEN_EXPIRE_DAYS = 7

pwd_context = CryptContext(schemes=['bcrypt'], deprecated='auto')
oauth2_scheme = OAuth2PasswordBearer(tokenUrl='/api/auth/login')


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    if 'sub' in to_encode and not isinstance(to_encode['sub'], str):
        to_encode['sub'] = str(to_encode['sub'])
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({'exp': expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def create_refresh_token(user_id: int):
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    token = jwt.encode({'user_id': user_id, 'exp': expire}, SECRET_KEY, algorithm=ALGORITHM)
    return token, expire


def get_user_by_email(db: Session, email: str):
    return db.query(models.User).filter(models.User.email == email).first()


def authenticate_user(db: Session, email: str, password: str):
    user = get_user_by_email(db, email)
    if not user or not verify_password(password, user.password_hash):
        return None
    return user


def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail='Could not validate credentials',
        headers={'WWW-Authenticate': 'Bearer'},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id_raw = payload.get('sub')
        if user_id_raw is None:
            raise credentials_exception
        user_id = int(user_id_raw)
    except (JWTError, ValueError, TypeError):
        raise credentials_exception
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if user is None:
        raise credentials_exception
    return user


def create_user(db: Session, user_data):
    user = models.User(
        name=user_data.name,
        email=user_data.email,
        password_hash=get_password_hash(user_data.password),
        age=user_data.age,
        experience_level=user_data.experience_level,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user
