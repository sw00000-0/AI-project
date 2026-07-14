# AIArtTeacher

## Setup

1. Create a virtual environment and activate it.
2. Install dependencies:
   ```bash
   pip install fastapi uvicorn sqlalchemy pydantic passlib[bcrypt] python-jose google-generativeai
   ```
3. Copy `.env.sample` from the project root to `.env` and set `GEMINI_API_KEY` and optional `GEMINI_MODEL`.
4. Export optional DB URL:
   ```bash
   export AIAT_DATABASE_URL=sqlite:///./aiartteacher.db
   ```
5. Initialize the database by importing the package or by starting the app.

## Run

```bash
uvicorn AIArtTeacher.main:app --reload --host 0.0.0.0 --port 8000
```

## Core API

- `POST /api/auth/signup`
- `POST /api/auth/login`
- `GET /api/uploads/signed-url?filename=...&content_type=...`
- `POST /api/submissions`
- `GET /api/submissions/{submission_id}`
- `POST /api/submissions/{submission_id}/feedback`
- `POST /api/communities`
- `POST /api/challenges/{month}/entries`
- `POST /api/orientation-prompts/{prompt_id}/response`
