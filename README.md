---
title: GapFolio
emoji: 📄
colorFrom: green
colorTo: gray
sdk: docker
app_port: 8000
pinned: false
---

# GapFolio

Resume-to-job-description gap analysis. Upload a resume, paste a job
description, and get a match score with evidence for every requirement, a
separate ATS-readability check, and a ranked roadmap of what to fix first.

- `gapfolio/` — FastAPI backend, local SBERT matching, tests, evaluation scripts
  (see `gapfolio/README.md`)
- `gapfolio-ui/` — React dashboard, built into the same container
- `Dockerfile` — one image that serves the API, the model and the dashboard

This is a classroom demo. Data is not persisted across restarts: please do not
upload real personal resumes.
