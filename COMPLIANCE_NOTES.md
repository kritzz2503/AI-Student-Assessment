# Security & Compliance Notes

This document is an honest account of what this project does and does not
satisfy with respect to data security and educational privacy regulations
(FERPA, GDPR). It is written for a student project / prototype context,
not as a legal compliance certification — no software can self-certify as
"FERPA compliant" or "GDPR compliant"; that requires a real audit, a data
processing agreement, and organizational policy, none of which a codebase
alone can provide.

---

## 1. What this project DOES do

### 1.1 Encryption at rest (partial)
- Student answers (`student_answer` column in `submissions`) are encrypted
  before being written to the SQLite database, using `Fernet` symmetric
  encryption from Python's `cryptography` library (`security.py`).
- The encryption key is stored in `secret.key`, generated automatically on
  first run. This file must be treated like a password: never committed to
  version control, never zipped into a submission, never shared.
- Decryption is "tolerant": if a stored value isn't actually encrypted
  (e.g. data saved before this feature existed), it's returned as-is
  instead of crashing. This means **older rows in an existing database
  remain in plain text** unless manually re-encrypted — encryption only
  applies going forward from when this feature was added.

### 1.2 What is intentionally NOT encrypted
- `student_name` is stored in plain text. This is a deliberate design
  choice, not an oversight: the system uses exact-match SQL lookups
  (`WHERE student_name = ?`, `GROUP BY student_name`) for plagiarism
  checks, inconsistency checks, and analytics. Fernet encryption is
  non-deterministic (the same input produces different ciphertext each
  time), so encrypting this field would break every one of those lookups.
  A production system solving this properly would use a separate students
  table with a numeric ID, and either leave names in plain text (treating
  them like a username) or use deterministic/searchable encryption —
  both larger changes than fit this project's scope.
- The `feedback` column (JSON blob of AI-generated feedback) is not
  encrypted.
- The `.env` file (API keys) is not encrypted — it relies on file-system
  permissions and not being committed/shared, same as standard practice
  for most small projects. **This already proved to be a real risk once
  in this project** — an earlier zip submission included a live API key.
  Always exclude `.env` and `secret.key` from anything zipped or pushed.

### 1.3 Multi-institution data isolation (Scalability-related, not strictly
a privacy requirement, but relevant to data segregation)
- Exams belong to an `institution_id`. Analytics queries can be filtered
  by institution name, so one deployment can serve multiple
  schools/colleges without their data appearing mixed together in reports.
- This is **application-level logical isolation only** — all institutions
  still share the same physical database file. A real multi-tenant
  system handling regulated data would typically use separate databases
  or stronger row-level security guarantees per tenant.

---

## 2. What this project does NOT do (be upfront about these)

- **No encryption in transit.** The Flask app runs on plain HTTP
  (`http://localhost:5000`), not HTTPS. Anyone on the same network could
  intercept requests. Production deployment would need TLS.
- **No authentication or access control.** Any caller can hit any API
  route — there's no login, no per-teacher or per-institution access
  restriction enforced at the API layer (the institution filter on
  `/analytics` is opt-in via a query parameter, not an enforced boundary).
  A real system would need user accounts, roles (teacher/student/admin),
  and authorization checks on every route.
- **No audit logging.** There's no record of who accessed which student's
  data, when, or why — both FERPA and GDPR generally expect some form of
  access accountability for personal data.
- **No data retention or deletion policy.** Data is kept indefinitely.
  GDPR's "right to erasure" and FERPA's record-retention expectations
  both require a defined policy and a mechanism to act on it (e.g. a
  "delete my data" request) — neither exists here.
- **No consent flow.** Student data is processed without any explicit
  consent capture step.
- **No data processing agreement / vendor review.** This project sends
  student answers to a third-party API (Groq, for feedback generation
  and question generation). Real compliance would require reviewing
  that vendor's own data handling terms and likely a formal agreement —
  this has not been done.
- **Key management is minimal.** `secret.key` lives as a plain file next
  to the code. Production systems typically use a managed secrets
  service (e.g. a cloud KMS) rather than a local key file.

---

## 3. Honest summary for a report or viva

> "This project implements column-level encryption for sensitive answer
> data at rest, and supports logical data isolation across institutions
> at the database level. These are genuine security improvements over the
> original prototype. However, the project does not implement
> authentication, transport encryption (HTTPS), audit logging, consent
> management, or a data retention policy — all of which would be required
> for actual FERPA/GDPR compliance in a real deployment. We treat this as
> a security-aware prototype, not a compliant production system."

This framing is more defensible in a viva than claiming "compliance" —
an examiner asking "is student name encrypted?" or "is there a login
system?" will quickly find the real gaps if those are overclaimed.
