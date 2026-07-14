import os
import uuid
from typing import Dict, Tuple
from urllib.parse import urlparse

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'webp', 'pdf'}
MAX_UPLOAD_MB = int(os.getenv('AIAT_MAX_UPLOAD_MB', '25'))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024


def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def generate_signed_upload_url(filename: str, content_type: str) -> dict:
    key = f'uploads/{uuid.uuid4().hex}/{filename}'
    return {
        'url': f'https://example-s3.local/{key}',
        'fields': {},
        'key': key,
        'max_bytes': MAX_UPLOAD_BYTES,
        'allowed_types': list(ALLOWED_EXTENSIONS),
    }


def normalize_image_metadata(path: str, width: int | None = None, height: int | None = None, fmt: str | None = None) -> Dict[str, str | int]:
    return {
        'path': path,
        'width': width or 0,
        'height': height or 0,
        'format': fmt or 'unknown',
    }


def compute_overall_score(scores: dict) -> int:
    color = scores.get('color_contrast', 0)
    light = scores.get('light_shadow', 0)
    sym = scores.get('symmetry', 0)
    return int(round(color * 0.35 + light * 0.35 + sym * 0.30))
