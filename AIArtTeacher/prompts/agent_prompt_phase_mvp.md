# Agent Instruction — Phase‑Oriented MVP‑First

Purpose: Drive an agent to generate Phase One artifacts for "AI Art Teacher", then stop for human acceptance before Phase Two.

Top-level requirements:
- Complete Phase One before any Phase Two work.
- Strictly follow ProjectDetails constraints (auth fields, models, scoring weights, orientation prompt rules).

Subtasks:
1. Database & Models
   - Create DB migrations for `User`, `Submission`, `Feedback`, `Community`, `Challenge`.
   - Add model field types exactly as specified.

2. Authentication
   - Implement signup/login endpoints: POST `/api/auth/signup`, POST `/api/auth/login`.
   - Enforce required signup fields: `name` (string), `age` (integer), `experience_level` (enum: beginner|intermediate|advanced).

3. File Upload Flow
   - Implement signed-upload URL generation and acceptance of JPEG/PNG/WebP/PDF up to 25MB.
   - Save original image URL on `Submission` record; store annotated images separately.

4. Orientation Detection
   - Asynchronously check orientation on upload; if rotated (>90% confidence) set `is_upside_down=true` and create a pending prompt for the user.
   - Do NOT auto-rotate. Provide API/UI prompt: "Detected rotated image — keep orientation as-is or auto-rotate?" and accept user note explaining creative intent.

5. LLM Feedback Integration
   - Provide two LLM prompt templates (realistic and creative) with placeholders `{experience_level}`, `{submission_metadata}`, `{reference_image_url}`, `{previous_feedback}`.
   - Implement automatic realistic-mode feedback generation producing numeric `color_contrast`, `light_shadow`, `symmetry` (0–100) and compute `overall_score` as weighted average: 35% color_contrast, 35% light_shadow, 30% symmetry.

6. API Endpoints
   - Implement core endpoints from ProjectDetails (create submission, fetch submission+feedback, post feedback, create community, challenge entry).

7. Frontend MVP
   - Create 3‑pane layout with components: SubmissionHistoryList (left), SubmissionViewer (center), FeedbackPanel (right, read-only annotator display).

8. Tests & CI
   - Unit tests for signup/login, create submission, fetch submission+feedback, challenge opt-in, create community, delete submission.
   - Integration test for upload flow including orientation detection prompt.

Verification steps (explicit):
- DB & Models: run migration command and confirm schema contains exact fields. Example (replace with your DB tool):

```bash
# run migrations
alembic upgrade head

# verify schema contains 'is_upside_down' on submissions
psql $DB_URL -c "\d+ submissions"
```

- Auth endpoints: curl signup/login and validate responses and stored `experience_level`:

```bash
curl -X POST /api/auth/signup -H 'Content-Type: application/json' -d '{"name":"A","email":"a@example.com","password":"P@ssw0rd","age":20,"experience_level":"advanced"}'
# expect 201 and JSON user with id and created_at
```

- Upload + Orientation: simulate signed upload, POST submission, do not be fixated by upside down - it is just an example. assert `is_upside_down` true when detector confidence >90% and that no image bytes were altered on server. Also verify prompt record exists for user action. 

- LLM feedback: after automatic feedback run, GET `/api/submissions/:id` and assert feedback contains numeric `color_contrast`, `light_shadow`, `symmetry`, and computed `overall_score` matching weighting formula within rounding tolerance.

- Frontend: load main view and confirm left panel lists submissions, center displays image, right shows feedback sections collapsed by metric.

Acceptance gate: produce a one‑page acceptance checklist mapping each subtask to a passing test and require human sign-off before Phase Two is generated.

Take inspiration from CoArtist website which has the following aspects:
- AI‑Driven Feedback: Extend the realistic LLM template to explicitly evaluate composition, color harmony, and anatomical precision; include a concise "senior-artist" style critique section. Subtasks: add prompt fields, IR mock responses, and unit tests asserting presence of composition/color/anatomy keys in LLM output.
- Visual Next‑Steps: Support generation of non‑destructive overlay metadata (numbered callouts with normalized coordinates and suggested pixel-level adjustments). Subtasks: define overlay JSON schema, pipeline to generate and store annotated overlay images, and a rendering verification test that maps callouts to `referenced_labels` integers.
- Rapid Studio Upload (optional): Accept PSD/RAW metadata fields and optimize signed‑upload flow for large high‑res files; include client hint for drag‑drop UX. Verification: simulate PSD metadata upload and confirm no image truncation and storage of original file URL.
- Growth Analytics: Add optional analytics tracker that records per‑submission metric history (date, each numeric metric) and an endpoint to retrieve time series for a user. Verification: ingest synthetic feedback data and confirm analytics endpoint returns correct aggregates and basic trend detection.
- Mobile Quick Critique (CoArtist‑like): Provide an optional lightweight API path that accepts a single image and returns a rapid scored feedback snippet (no account required for the stub). Subtasks: implement rate limiting, stubbed LLM quick reply, and a test confirming sub‑5s response under mocked LLM latency.

Verification notes for optional features: treat these as enhancers — include feature flags and ensure all additions remain opt‑in. Do not modify core scoring rules or orientation behavior when enabling these suggestions.
