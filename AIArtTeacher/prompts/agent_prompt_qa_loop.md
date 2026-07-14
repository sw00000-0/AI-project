# Agent Instruction — Continuous QA + Reviewer Loop

Purpose: Produce code iteratively while running automated QA, security checks, and generating PR-style summaries requiring human sign-off for schema changes.

Loop rules:
- For every iteration: generate code changes, run unit/integration tests, run static analysis/security checks, and produce a test/security report.
- Auto-apply trivial fixes (formatting, import ordering, small failing tests with safe changes). For non-trivial failures, create a human-review patch and clearly list blockers.
- Require explicit approval before: schema-breaking migrations, Phase transitions, or enabling advanced scoring visibility.

Subtasks per iteration:
1. Generate code change
   - Produce a focused patch and list of modified files.
   - Run local tests immediately.

2. Automated checks
   - Run unit tests (`pytest`), run linter (`flake8`/`eslint`), run formatter check (`black --check`/`prettier --check`).
   - Run security checks: validate input schemas, file upload size/type limits, RBAC enforcement on private threads, and an image-scan stub that flags disallowed content.

3. Report
   - Produce a concise report with: tests passed/failed, coverage delta, linter errors, security findings, and exact failing assertions.
   - If failures are trivial, auto-produce a follow-up patch that fixes them and re-run tests.

4. PR Summary for human review
   - Include: intention, changed files, migration steps, new env vars, sample curl flows, acceptance criteria mapping to ProjectDetails, and reviewer questions.

Verification steps (explicit):
- Run full test suite: `pytest --maxfail=1 --disable-warnings -q` and attach command output.
- Lint/format checks: `black --check .` and `flake8 .` (or equivalent); show commands and exit codes.
- Security checks: run input validation unit tests and demonstrate RBAC enforcement by attempting unauthorized actions and expecting 403 responses.

Gate rules:
- Schema-changing migrations must be accompanied by a reversible migration and a human sign-off file that lists the rationale.
- Phase transitions require all Phase One acceptance tests to pass and a signed acceptance checklist.

Deliverable: for each iteration, produce a PR-style artifact (patch + report + acceptance checklist). Stop when human reviewer approves the PR.

Optional CoArtist‑style QA checks and metrics:
- Visual Next‑Steps QA: include a rendering verification step that checks overlay generation correctness and that numbered labels map to `referenced_labels` integers.
- Growth Analytics QA: include analytics smoke tests that verify time‑series endpoints return increasing/decreasing trends for synthetic user data and that the automated ledger entries are persisted.
- Mobile Quick Critique QA: include latency benchmarks for the rapid critique endpoint (mocked LLM) ensuring responses under configurable threshold (e.g., 5s) and rate‑limit enforcement tests.

Security reminder: optional features must not bypass authentication or expose private submissions. All mobile/quick‑critique stubs must be rate‑limited and monitored.
