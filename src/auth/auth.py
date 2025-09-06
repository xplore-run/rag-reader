"""Authentication logic using JWT."""

import os
from datetime import datetime, timedelta
from typing import Optional

from jose import JWTError, jwt
from passlib.context import CryptContext
from dotenv import load_dotenv

from ..utils.logger import logger

# Load environment variables
load_dotenv()

# Configuration
SECRET_KEY = os.getenv("JWT_SECRET", "your-secret-key-here")
ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_HOURS = int(os.getenv("JWT_EXPIRATION_HOURS", "24"))

# Hardcoded user credentials from environment
VALID_USER_EMAIL = os.getenv("AUTH_USER_EMAIL", "xyz@abc.com")
VALID_USER_PASSWORD = os.getenv("AUTH_USER_PASSWORD", "test@123#")

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password."""
    return pwd_context.hash(password)


def authenticate_user(email: str, password: str) -> bool:
    """Authenticate user with hardcoded credentials."""
    if email != VALID_USER_EMAIL:
        logger.warning(f"Login attempt with invalid email: {email}")
        return False
    
    if password != VALID_USER_PASSWORD:
        logger.warning(f"Login attempt with invalid password for email: {email}")
        return False
    
    logger.info(f"Successful login for email: {email}")
    return True


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token."""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    
    return encoded_jwt


def decode_access_token(token: str) -> Optional[str]:
    """Decode and validate a JWT access token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        
        if email is None:
            return None
            
        return email
        
    except JWTError as e:
        logger.error(f"JWT decode error: {str(e)}")
        return None