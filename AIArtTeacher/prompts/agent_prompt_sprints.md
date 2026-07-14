# Agent Instruction — Sprint‑by‑Sprint Feature Increments

Purpose: Drive the agent through short, focused sprints delivering one small, testable artifact per sprint and producing runnable patches and tests.

Sprint rules:
- Each sprint must target exactly one deliverable: one model+migration, one API endpoint, one frontend component, one upload flow piece, or one test.
- Each sprint must include: code diff (patch), unit tests that pass locally, example API calls, and one review question.
- Run tests and fix obvious failures within the same sprint.

Subtasks (sample sprint sequence):
Sprint 1 — Auth Model & Signup Endpoint
  - Create `User` migration and model (fields per spec).
  - Implement POST `/api/auth/signup` handler and validation.
  - Tests: unit tests for signup input validation and password hashing.
  - Verification: run `pytest tests/test_auth.py::test_signup` and expect pass.

Sprint 2 — Signed Upload URL + Submission Model
  - Implement signed upload generation endpoint and `Submission` model/migration.
  - Tests: simulate signed upload flow with mocked storage and assert submission record created with `original_image_url`.
  - Verification: `pytest tests/test_upload.py::test_signed_upload_flow`.

Sprint 3 — Orientation Check Hook
  - Add asynchronous orientation job stub and set `is_upside_down` when confidence>90%.
  - Tests: unit test the orientation decision logic with sample images or mocked detector.
  - Verification: `pytest tests/test_orientation.py::test_upside_down_flag`.

Sprint 4 — Realistic Feedback Generator
  - Implement the backend job that calls LLM with realistic template and saves `Feedback` with numeric metrics and computed `overall_score`.
  - Tests: mock LLM response and assert correct score computation.
  - Verification: `pytest tests/test_feedback.py::test_realistic_score_weights`.

Sprint output requirements (every sprint):
- Provide a git-style patch or apply_patch-ready diff.
- Include test command(s) and expected output snippet.
- One short reviewer question: e.g., "Should password policy require 10+ chars?"

Verification checklist (per sprint):
- Code compiles and tests pass locally: run `pytest` and confirm exit code 0.
- Linter/formatter: run `black --check .` and `flake8` (or project equivalents).
- API smoke test: curl the new endpoint and confirm HTTP success and JSON schema match.

Stop condition: after completing the prioritized sprints (auth, upload+orientation, realistic feedback, submission listing UI, feedback viewer UI), pause and present a sprint summary and ask for approval to continue.

Optional CoArtist‑inspired sprint extensions:
- Sprint: AI‑Driven Feedback prompt enhancement
  - Subtask: update realistic LLM prompt to include composition, color harmony, and anatomy checks; add unit tests validating presence of those critique items in responses.
  - Verification: mock LLM and assert keys `composition`, `color_harmony`, `anatomical_precision` exist.
- Sprint: Visual Next‑Steps overlay schema
  - Subtask: design overlay JSON and a small rendering helper for numbered non‑destructive overlays; produce a test rendering a sample overlay to PNG.
  - Verification: generated overlay PNG includes numbered markers and matches `referenced_labels` positions within tolerance.
- Sprint: Rapid Studio Upload support
  - Subtask: add support for PSD/RAW metadata and increase upload chunking test coverage for large files.
  - Verification: run integration upload with a large test file and assert successful storage and metadata preservation.

Note: These sprints are optional extensions and should be scheduled only after Phase One acceptance unless explicitly prioritized.
