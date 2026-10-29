import React, { useState, useRef, useEffect } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  ResponsiveContainer,
  Cell,
  Tooltip,
} from "recharts";
import * as api from "./api.js";

/* ------------------------------------------------------------------ */
/*  GapFolio — resume → job-fit gap analysis                           */
/*  Design language: a marked-up document. Marker = proven,            */
/*  amber underline = partial, redline = missing.                      */
/* ------------------------------------------------------------------ */

const CSS = `
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,400;12..96,600;12..96,800&family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

.gf {
  --ink: #12151a;
  --paper: #edefe9;
  --card: #ffffff;
  --graphite: #5c6672;
  --rule: #d7dbd1;
  --marker: #cdeb45;
  --marker-soft: #eaf7b0;
  --redline: #e0402f;
  --redline-soft: #fadfda;
  --amber: #d98a1f;
  --amber-soft: #f8e9cd;
  --panel: #171b21;

  font-family: 'IBM Plex Sans', ui-sans-serif, system-ui, sans-serif;
  color: var(--ink);
  background: var(--paper);
  min-height: 100vh;
  line-height: 1.5;
  -webkit-font-smoothing: antialiased;
}
.gf *, .gf *::before, .gf *::after { box-sizing: border-box; }
.gf button { font: inherit; color: inherit; cursor: pointer; }
.gf :focus-visible { outline: 2px solid var(--ink); outline-offset: 2px; }

.gf-display { font-family: 'Bricolage Grotesque', 'IBM Plex Sans', sans-serif; letter-spacing: -0.035em; line-height: 1.02; font-weight: 800; }
.gf-mono { font-family: 'IBM Plex Mono', ui-monospace, monospace; }
.gf-eyebrow { font-family: 'IBM Plex Mono', monospace; font-size: 10px; letter-spacing: 0.16em; text-transform: uppercase; color: var(--graphite); font-weight: 500; }

.gf-shell { max-width: 1180px; margin: 0 auto; padding: 0 20px 80px; }

/* masthead ------------------------------------------------------- */
.gf-top { display: flex; align-items: center; justify-content: space-between; gap: 16px; padding: 20px 0 18px; border-bottom: 1px solid var(--rule); flex-wrap: wrap; }
.gf-wordmark { font-family: 'Bricolage Grotesque', sans-serif; font-weight: 800; font-size: 22px; letter-spacing: -0.05em; display: flex; align-items: baseline; gap: 1px; }
.gf-wordmark .hl { background: var(--marker); padding: 0 3px; box-decoration-break: clone; }
.gf-tabs { display: flex; gap: 4px; background: #e2e5dc; padding: 3px; border-radius: 2px; }
.gf .gf-tab { border: 0; background: transparent; padding: 7px 14px; font-size: 13px; font-weight: 500; border-radius: 1px; color: var(--graphite); }
.gf .gf-tab[data-on="true"] { background: var(--card); color: var(--ink); box-shadow: 0 1px 2px rgba(0,0,0,.07); }

/* hero ----------------------------------------------------------- */
.gf-hero { padding: 44px 0 30px; max-width: 760px; }
.gf-hero h1 { font-size: clamp(38px, 6.4vw, 68px); margin: 12px 0 0; }
.gf-hero h1 .strike { position: relative; white-space: nowrap; }
.gf-hero h1 .strike::after { content: ""; position: absolute; left: -2%; right: -2%; top: 52%; height: 5px; background: var(--redline); transform: rotate(-1.4deg); }
.gf-hero p { font-size: 17px; color: var(--graphite); max-width: 560px; margin: 20px 0 0; }

.gf-legend { display: flex; gap: 18px; flex-wrap: wrap; margin-top: 26px; }
.gf-legend span { display: inline-flex; align-items: center; gap: 7px; font-size: 12px; font-family: 'IBM Plex Mono', monospace; color: var(--graphite); }
.gf-swatch { width: 22px; height: 10px; display: inline-block; }

/* cards ---------------------------------------------------------- */
.gf-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 10px; }
@media (max-width: 860px) { .gf-grid { grid-template-columns: 1fr; } }
.gf-card { background: var(--card); border: 1px solid var(--rule); padding: 18px; }
.gf-card-head { display: flex; align-items: center; justify-content: space-between; gap: 10px; margin-bottom: 12px; }
.gf-card h3 { font-family: 'Bricolage Grotesque', sans-serif; font-weight: 600; font-size: 16px; letter-spacing: -0.02em; margin: 0; }
.gf-ta { width: 100%; min-height: 210px; resize: vertical; border: 1px solid var(--rule); background: #fcfdfb; padding: 12px; font-family: 'IBM Plex Mono', monospace; font-size: 12.5px; line-height: 1.65; color: var(--ink); }
.gf-ta::placeholder { color: #9aa3ac; }

/* These need to out-specify the '.gf button { color: inherit }' reset above —
   at equal weight the reset wins and a primary button renders ink-on-ink. */
.gf .gf-btn { border: 1px solid var(--ink); background: var(--ink); color: #fff; padding: 12px 22px; font-weight: 600; font-size: 14px; display: inline-flex; align-items: center; gap: 10px; }
.gf .gf-btn:disabled { opacity: .4; cursor: not-allowed; }
.gf .gf-btn-ghost { background: transparent; color: var(--ink); border: 1px solid var(--rule); padding: 7px 12px; font-size: 12px; font-weight: 500; }
.gf .gf-btn-ghost:hover { border-color: var(--ink); }
.gf-actions { display: flex; align-items: center; gap: 14px; margin-top: 18px; flex-wrap: wrap; }
.gf-note { font-size: 12px; color: var(--graphite); }

/* scan ----------------------------------------------------------- */
.gf-scan { background: var(--panel); color: #e8ebe4; padding: 26px; margin-top: 18px; position: relative; overflow: hidden; }
.gf-scan-line { position: absolute; left: 0; right: 0; height: 2px; background: var(--marker); box-shadow: 0 0 24px 6px rgba(205,235,69,.35); animation: gfsweep 1.9s cubic-bezier(.4,0,.6,1) infinite; }
@keyframes gfsweep { 0% { top: 0; } 100% { top: 100%; } }
.gf-stages { display: grid; gap: 9px; margin-top: 16px; }
.gf-stage { display: flex; align-items: center; gap: 12px; font-family: 'IBM Plex Mono', monospace; font-size: 12.5px; color: #7f8a96; }
.gf-stage[data-on="true"] { color: #e8ebe4; }
.gf-stage[data-done="true"] { color: var(--marker); }
.gf-dot { width: 7px; height: 7px; border: 1px solid currentColor; flex: none; }
.gf-stage[data-done="true"] .gf-dot { background: var(--marker); border-color: var(--marker); }
@media (prefers-reduced-motion: reduce) { .gf-scan-line { animation: none; top: 50%; } }

/* score ---------------------------------------------------------- */
.gf-score { background: var(--panel); color: #f3f5ef; padding: 30px; display: grid; grid-template-columns: auto 1fr; gap: 34px; align-items: center; }
@media (max-width: 720px) { .gf-score { grid-template-columns: 1fr; gap: 20px; } }
.gf-score-num { font-family: 'Bricolage Grotesque', sans-serif; font-weight: 800; font-size: 92px; line-height: .82; letter-spacing: -0.06em; }
.gf-score-num sup { font-size: 22px; vertical-align: super; margin-left: 2px; opacity: .5; letter-spacing: 0; }
.gf-verdict { font-family: 'Bricolage Grotesque', sans-serif; font-size: 21px; font-weight: 600; letter-spacing: -0.02em; margin: 0 0 6px; }
.gf-score p { color: #a8b2bc; font-size: 14px; margin: 0; max-width: 52ch; }
.gf-coverage { display: flex; gap: 3px; margin-top: 18px; }
.gf-coverage i { height: 20px; flex: 1; }
.gf-cov-key { display: flex; gap: 16px; margin-top: 9px; font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: #7f8a96; }

/* marked-up JD --------------------------------------------------- */
.gf-jd { background: var(--card); border: 1px solid var(--rule); padding: 22px 24px; font-size: 14.5px; line-height: 2.05; white-space: pre-wrap; max-height: 460px; overflow: auto; }
.gf-mk { padding: 2px 2px; cursor: pointer; border: 0; font: inherit; text-align: left; }
.gf-mk-proven { background: var(--marker); }
.gf-mk-partial { background: var(--amber-soft); border-bottom: 2px solid var(--amber); }
.gf-mk-missing { background: var(--redline-soft); text-decoration: line-through; text-decoration-color: var(--redline); text-decoration-thickness: 2px; }
.gf-mk[data-sel="true"] { box-shadow: 0 0 0 2px var(--ink); }

/* ledger --------------------------------------------------------- */
.gf-row { border-bottom: 1px solid var(--rule); padding: 14px 0; display: grid; grid-template-columns: 26px 1fr auto; gap: 14px; align-items: start; }
.gf-row:last-child { border-bottom: 0; }
.gf-idx { font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: var(--graphite); padding-top: 3px; }
.gf-req { font-weight: 600; font-size: 14.5px; }
.gf-ev { font-size: 13px; color: var(--graphite); margin-top: 3px; }
.gf-pill { font-family: 'IBM Plex Mono', monospace; font-size: 10px; letter-spacing: .1em; text-transform: uppercase; padding: 4px 8px; white-space: nowrap; font-weight: 500; }
.gf-pill-proven { background: var(--marker); }
.gf-pill-partial { background: var(--amber-soft); color: #7a4d05; }
.gf-pill-missing { background: var(--redline-soft); color: #8e2114; }

/* fixes ---------------------------------------------------------- */
.gf-fix { border-left: 3px solid var(--redline); background: var(--card); border-top: 1px solid var(--rule); border-right: 1px solid var(--rule); border-bottom: 1px solid var(--rule); padding: 16px 18px; }
.gf-fix[data-p="medium"] { border-left-color: var(--amber); }
.gf-fix-top { display: flex; align-items: baseline; justify-content: space-between; gap: 12px; flex-wrap: wrap; }
.gf-fix h4 { margin: 0; font-size: 15px; font-family: 'Bricolage Grotesque', sans-serif; font-weight: 600; letter-spacing: -0.02em; }
.gf-fix p { margin: 8px 0 0; font-size: 13.5px; color: #39414b; }
.gf-proof { margin-top: 10px; padding: 9px 11px; background: var(--marker-soft); font-family: 'IBM Plex Mono', monospace; font-size: 12px; line-height: 1.6; }

.gf-diff { border: 1px solid var(--rule); background: var(--card); }
.gf-diff div { padding: 12px 14px; font-size: 13.5px; }
.gf-diff .old { color: #8e2114; background: var(--redline-soft); text-decoration: line-through; text-decoration-color: rgba(142,33,20,.4); }
.gf-diff .new { border-top: 1px solid var(--rule); }
.gf-diff .why { border-top: 1px solid var(--rule); font-size: 12px; color: var(--graphite); background: #fafbf8; }

.gf-check { display: flex; align-items: flex-start; gap: 11px; padding: 11px 0; border-bottom: 1px dashed var(--rule); }
.gf-check:last-child { border-bottom: 0; }
.gf-chip { width: 16px; height: 16px; flex: none; margin-top: 2px; display: grid; place-items: center; font-size: 10px; font-weight: 700; color: #fff; font-family: 'IBM Plex Mono', monospace; }

.gf-sec { margin-top: 44px; }
.gf-sec-head { display: flex; align-items: baseline; gap: 14px; border-bottom: 2px solid var(--ink); padding-bottom: 8px; margin-bottom: 18px; }
.gf-sec-head h2 { font-family: 'Bricolage Grotesque', sans-serif; font-size: 22px; font-weight: 700; letter-spacing: -0.03em; margin: 0; }
.gf-sec-head span { margin-left: auto; }

.gf-stack { display: grid; gap: 10px; }
.gf-two { display: grid; grid-template-columns: 1.35fr 1fr; gap: 16px; align-items: start; }
@media (max-width: 900px) { .gf-two { grid-template-columns: 1fr; } }

/* batch ---------------------------------------------------------- */
.gf-kpis { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
@media (max-width: 780px) { .gf-kpis { grid-template-columns: repeat(2, 1fr); } }
.gf-kpi { background: var(--card); border: 1px solid var(--rule); padding: 16px; }
.gf-kpi b { display: block; font-family: 'Bricolage Grotesque', sans-serif; font-size: 34px; font-weight: 800; letter-spacing: -0.04em; line-height: 1.1; margin-top: 6px; }
.gf-table { width: 100%; border-collapse: collapse; background: var(--card); border: 1px solid var(--rule); }
.gf-table th { text-align: left; font-family: 'IBM Plex Mono', monospace; font-size: 10px; letter-spacing: .12em; text-transform: uppercase; color: var(--graphite); font-weight: 500; padding: 11px 14px; border-bottom: 1px solid var(--rule); }
.gf-table td { padding: 12px 14px; border-bottom: 1px solid #eef0ea; font-size: 13.5px; }
.gf-table tr:last-child td { border-bottom: 0; }
.gf-bar { height: 6px; background: #e6e9e0; width: 90px; display: inline-block; vertical-align: middle; margin-right: 9px; }
.gf-bar i { display: block; height: 100%; }

.gf-err { background: var(--redline-soft); border: 1px solid var(--redline); padding: 14px 16px; font-size: 13.5px; margin-top: 16px; }
.gf-foot { margin-top: 60px; padding-top: 20px; border-top: 1px solid var(--rule); font-size: 12px; color: var(--graphite); display: flex; justify-content: space-between; gap: 16px; flex-wrap: wrap; }

/* account + gates ------------------------------------------------- */
.gf-acct { display: flex; align-items: center; gap: 10px; font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: var(--graphite); }
.gf-overlay { position: fixed; inset: 0; background: rgba(18,21,26,.55); display: grid; place-items: center; padding: 20px; z-index: 50; overflow: auto; }
.gf-sheet { background: var(--card); border: 1px solid var(--ink); width: 100%; max-width: 520px; padding: 26px 28px; }
.gf-sheet h3 { font-family: 'Bricolage Grotesque', sans-serif; font-size: 24px; font-weight: 800; letter-spacing: -0.03em; margin: 8px 0 6px; }
.gf-sheet > p { font-size: 13.5px; color: var(--graphite); margin: 0 0 18px; }
.gf-field { display: block; margin-bottom: 12px; }
.gf-field span { display: block; font-family: 'IBM Plex Mono', monospace; font-size: 10px; letter-spacing: .14em; text-transform: uppercase; color: var(--graphite); margin-bottom: 5px; }
.gf-input { width: 100%; border: 1px solid var(--rule); background: #fcfdfb; padding: 10px 12px; font-family: 'IBM Plex Mono', monospace; font-size: 13px; color: var(--ink); }
.gf-input:focus { outline: 2px solid var(--ink); outline-offset: -1px; }
.gf-consent { border: 1px solid var(--rule); padding: 12px 14px; margin-bottom: 10px; display: flex; gap: 12px; align-items: flex-start; }
.gf-consent[data-req="true"] { border-left: 3px solid var(--redline); }
.gf-consent input { margin-top: 3px; flex: none; width: 15px; height: 15px; accent-color: var(--ink); }
.gf-consent b { font-size: 13.5px; display: block; margin-bottom: 3px; }
.gf-consent p { margin: 0; font-size: 12.5px; color: var(--graphite); line-height: 1.5; }
.gf-procs { font-size: 11.5px; color: var(--graphite); font-family: 'IBM Plex Mono', monospace; line-height: 1.7; border-top: 1px dashed var(--rule); margin-top: 14px; padding-top: 12px; }
`;

/* ------------------------------------------------------------------ */
/*  Sample data — keeps the demo alive if the network doesn't          */
/* ------------------------------------------------------------------ */

const SAMPLE_RESUME = `ANANYA IYER
B.Tech Computer Science, 2027 | Pune | ananya.iyer@email.com

EDUCATION
Vishwakarma Institute of Technology, Pune — B.Tech CSE, CGPA 8.4/10

PROJECTS
Campus Food Ordering App
- Built a food ordering website for our college canteen using React
- Worked on the frontend and connected it to a database
- Handled login and signup for users

Movie Recommendation System
- Made a recommendation system in Python
- Used pandas and scikit-learn to build the model
- Got good accuracy on the test data

INTERNSHIP
Web Development Intern, Nexlify Solutions (May–July 2026)
- Helped the team with website tasks
- Fixed bugs reported by the senior developer
- Attended daily standup meetings

SKILLS
HTML, CSS, JavaScript, React, Python, C++, MySQL, Git

POSITIONS
Coordinator, Technical Club — organised a coding event with 120 participants`;

const SAMPLE_JD = `Software Engineer I — Backend (Fresher)
Bengaluru | Full-time

About the role
You'll join the payments platform team building the APIs that move money for millions of users. You will own small services end to end, from design through deployment.

What you'll do
- Design and build REST APIs in Node.js or Python that are reliable under load
- Write and optimise SQL queries against PostgreSQL, including query tuning
- Containerise services with Docker and ship them through CI/CD pipelines
- Write unit and integration tests; we expect meaningful test coverage on every PR
- Debug production issues using logs and monitoring dashboards
- Work with AWS services such as EC2, S3 and Lambda

What we're looking for
- Strong fundamentals in data structures and algorithms
- Hands-on experience with Git and code review workflows
- Ability to communicate technical decisions clearly in writing
- Bonus: exposure to system design, message queues like Kafka, or Redis caching`;

const SAMPLE_MATCH = {
  role_title: "Software Engineer I — Backend",
  seniority: "Entry level / fresher",
  match_score: 46,
  readiness: "Not yet",
  headline:
    "Frontend-heavy profile applying to a backend role — the fundamentals are there, the backend proof is not.",
  requirements: [
    { id: 1, label: "REST APIs in Node.js or Python", jd_quote: "Design and build REST APIs in Node.js or Python", status: "partial", evidence: "React app 'connected to a database' — no API layer shown", weight: "core" },
    { id: 2, label: "SQL and PostgreSQL tuning", jd_quote: "Write and optimise SQL queries against PostgreSQL", status: "partial", evidence: "MySQL listed as a skill; no query work described", weight: "core" },
    { id: 3, label: "Docker and CI/CD", jd_quote: "Containerise services with Docker and ship them through CI/CD pipelines", status: "missing", evidence: null, weight: "core" },
    { id: 4, label: "Testing discipline", jd_quote: "Write unit and integration tests", status: "missing", evidence: null, weight: "core" },
    { id: 5, label: "Production debugging", jd_quote: "Debug production issues using logs and monitoring dashboards", status: "partial", evidence: "Fixed bugs during internship — no logs or monitoring named", weight: "core" },
    { id: 6, label: "AWS services", jd_quote: "Work with AWS services such as EC2, S3 and Lambda", status: "missing", evidence: null, weight: "core" },
    { id: 7, label: "DSA fundamentals", jd_quote: "Strong fundamentals in data structures and algorithms", status: "partial", evidence: "CSE degree, C++ listed; no DSA evidence on the page", weight: "core" },
    { id: 8, label: "Git and code review", jd_quote: "Hands-on experience with Git and code review workflows", status: "proven", evidence: "Git listed; internship mentions senior-developer review loop", weight: "core" },
    { id: 9, label: "Written communication", jd_quote: "communicate technical decisions clearly in writing", status: "partial", evidence: "Club coordinator role implies it, resume doesn't show it", weight: "nice-to-have" },
    { id: 10, label: "Caching and queues", jd_quote: "message queues like Kafka, or Redis caching", status: "missing", evidence: null, weight: "nice-to-have" },
  ],
};

const SAMPLE_FIXES = {
  roadmap: [
    { skill: "Docker + CI/CD", priority: "high", effort: "1 week", fix_type: "project", action: "Containerise the food-ordering backend and add a GitHub Actions workflow that runs tests and builds the image on every push.", proof: "Dockerised a 3-service app and cut deploy time from 20 min manual to a 4-min automated pipeline" },
    { skill: "Backend API ownership", priority: "high", effort: "2 weeks", fix_type: "project", action: "Rebuild the canteen app's data layer as a documented REST API in Express with auth, pagination and error handling — then describe the API, not the UI.", proof: "Built 11 REST endpoints in Express serving 120+ campus users, with JWT auth and request validation" },
    { skill: "Testing", priority: "high", effort: "4 days", fix_type: "project", action: "Add Jest or pytest coverage to an existing project and put the coverage number on the resume.", proof: "Wrote 40+ unit and integration tests, taking coverage to 82%" },
    { skill: "PostgreSQL query work", priority: "medium", effort: "1 week", fix_type: "course", action: "Work through a SQL performance course, then migrate one project from MySQL to Postgres and record a before/after query time.", proof: "Cut a 1.4s report query to 90ms by adding a composite index" },
    { skill: "AWS basics", priority: "medium", effort: "1 week", fix_type: "course", action: "Deploy one project to EC2 with static assets on S3 — the cheapest credible AWS line on a fresher resume.", proof: "Deployed on EC2 with S3 asset hosting, serving 120 monthly users at under ₹400/month" },
  ],
  ats: [
    { check: "Contact and header parse cleanly", status: "pass", note: "Name, degree, city and email are in plain text lines — machine-readable." },
    { check: "Section headings are standard", status: "pass", note: "Education, Projects, Skills and Internship use conventional names." },
    { check: "No quantified outcomes", status: "fail", note: "Only one number appears in the whole document (120 participants). Recruiters scan for numbers first." },
    { check: "Weak verbs on 5 of 9 bullets", status: "fail", note: "'Helped', 'worked on', 'handled' and 'made' describe presence, not contribution." },
    { check: "Skills listed without depth signals", status: "warn", note: "A flat comma list gives the matcher no way to tell React from a one-weekend tutorial." },
    { check: "No links to code", status: "warn", note: "No GitHub or deployed-project URL for a role that reviews code." },
  ],
  rewrites: [
    { before: "Helped the team with website tasks", after: "Shipped 14 frontend tickets across 3 sprints in a 5-person team, closing the sprint backlog two days early", why: "'Helped' is unverifiable. Volume, team size and timeframe are all things a recruiter can picture." },
    { before: "Made a recommendation system in Python. Got good accuracy on the test data", after: "Built a collaborative-filtering recommender on 10k MovieLens ratings, reaching 0.89 RMSE — 14% better than a popularity baseline", why: "'Good accuracy' means nothing without a metric and a baseline to beat." },
    { before: "Built a food ordering website for our college canteen using React", after: "Built and deployed a canteen ordering app used by 120+ students, cutting average queue time from 12 to 4 minutes", why: "The JD hires for impact under load. Users and a time saved make the project real." },
  ],
  questions: [
    "Your resume says you connected the canteen app to a database — walk me through the schema and what you'd change now.",
    "You've written React but this role is backend. What have you built where you owned the server side end to end?",
    "How would you debug an API endpoint that's fine in testing but times out for 5% of production requests?",
  ],
};

const BATCH = [
  { name: "Rohit Sharma", branch: "CSE", score: 82, gap: "System design depth", apps: 9 },
  { name: "Ananya Iyer", branch: "CSE", score: 46, gap: "Docker, CI/CD, testing", apps: 4 },
  { name: "Farhan Qureshi", branch: "IT", score: 71, gap: "Quantified outcomes", apps: 7 },
  { name: "Meera Nair", branch: "CSE", score: 88, gap: "Kafka exposure", apps: 11 },
  { name: "Devansh Patil", branch: "ENTC", score: 38, gap: "No backend projects", apps: 2 },
  { name: "Sana Kulkarni", branch: "IT", score: 64, gap: "SQL tuning, AWS", apps: 6 },
  { name: "Aditya Rao", branch: "CSE", score: 29, gap: "Resume unparseable by ATS", apps: 1 },
  { name: "Priya Deshmukh", branch: "ENTC", score: 57, gap: "Weak verbs, no metrics", apps: 5 },
];

const DIST = [
  { band: "0–39", n: 41, color: "#e0402f" },
  { band: "40–59", n: 78, color: "#d98a1f" },
  { band: "60–79", n: 112, color: "#a8c72e" },
  { band: "80–100", n: 63, color: "#7fa310" },
];

/* ------------------------------------------------------------------ */
/*  Auth + consent gates                                               */
/*                                                                     */
/*  The backend enforces consent as a route dependency, so a resume    */
/*  cannot reach the analysis pipeline without a granted scope. These  */
/*  two sheets are the client half of that contract.                   */
/* ------------------------------------------------------------------ */

function AuthSheet({ onDone, onCancel }) {
  const [mode, setMode] = useState("register");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");

  const submit = async (e) => {
    e.preventDefault();
    setErr("");
    setBusy(true);
    try {
      const res =
        mode === "register"
          ? await api.register(email, password, name)
          : await api.login(email, password);
      api.setToken(res.access_token);
      onDone(res);
    } catch (e2) {
      setErr(e2.message);
      setBusy(false);
    }
  };

  return (
    <div className="gf-overlay" onMouseDown={(e) => e.target === e.currentTarget && onCancel()}>
      <form className="gf-sheet" onSubmit={submit}>
        <div className="gf-eyebrow">{mode === "register" ? "Create an account" : "Welcome back"}</div>
        <h3>Your reports need somewhere to live.</h3>
        <p>
          Analyses are saved against your account so you can track progress across applications. You
          choose what is stored on the next screen.
        </p>

        {mode === "register" && (
          <label className="gf-field">
            <span>Full name</span>
            <input className="gf-input" value={name} onChange={(e) => setName(e.target.value)} required />
          </label>
        )}
        <label className="gf-field">
          <span>Email</span>
          <input
            className="gf-input"
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
          />
        </label>
        <label className="gf-field">
          <span>Password</span>
          <input
            className="gf-input"
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            minLength={8}
            required
          />
        </label>

        {err && <div className="gf-err" style={{ marginTop: 4 }}>{err}</div>}

        <div className="gf-actions">
          <button className="gf-btn" type="submit" disabled={busy}>
            {busy ? "Working…" : mode === "register" ? "Create account" : "Sign in"}
          </button>
          <button
            className="gf-btn-ghost"
            type="button"
            onClick={() => { setErr(""); setMode(mode === "register" ? "login" : "register"); }}
          >
            {mode === "register" ? "I already have an account" : "Create one instead"}
          </button>
          <button className="gf-btn-ghost" type="button" onClick={onCancel}>Cancel</button>
        </div>
      </form>
    </div>
  );
}

function ConsentSheet({ onDone, onCancel }) {
  const [policy, setPolicy] = useState(null);
  const [picked, setPicked] = useState({});
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");

  useEffect(() => {
    api
      .getPolicy()
      .then((p) => {
        setPolicy(p);
        const init = {};
        p.scopes.forEach((s) => (init[s.scope] = !!s.required));
        setPicked(init);
      })
      .catch((e) => setErr(e.message));
  }, []);

  const submit = async () => {
    setErr("");
    setBusy(true);
    try {
      for (const s of policy.scopes) {
        await api.setConsent(s.scope, !!picked[s.scope]);
      }
      onDone();
    } catch (e2) {
      setErr(e2.message);
      setBusy(false);
    }
  };

  return (
    <div className="gf-overlay" onMouseDown={(e) => e.target === e.currentTarget && onCancel()}>
      <div className="gf-sheet">
        <div className="gf-eyebrow">
          What we do with your resume{policy ? ` · policy ${policy.version}` : ""}
        </div>
        <h3>You decide what's stored.</h3>
        <p>
          The first is required to run an analysis at all — we can't match a resume we're not allowed
          to read. The other two are genuinely optional and you can withdraw any of them later.
        </p>

        {!policy && !err && <p className="gf-note">Loading the policy…</p>}

        {policy &&
          policy.scopes.map((s) => (
            <label key={s.scope} className="gf-consent" data-req={s.required}>
              <input
                type="checkbox"
                checked={!!picked[s.scope]}
                disabled={s.required}
                onChange={(e) => setPicked({ ...picked, [s.scope]: e.target.checked })}
              />
              <div>
                <b>
                  {s.scope.replace(/_/g, " ")}
                  {s.required ? " — required" : " — optional"}
                </b>
                <p>{s.description}</p>
              </div>
            </label>
          ))}

        {policy && (
          <div className="gf-procs">
            Resume text is retained for {policy.retention_days} days after its last analysis.
            {policy.processors.map((p) => (
              <div key={p.name}>
                → {p.name}: {p.purpose} (retains: {p.retains})
              </div>
            ))}
          </div>
        )}

        {err && <div className="gf-err">{err}</div>}

        <div className="gf-actions">
          <button className="gf-btn" onClick={submit} disabled={!policy || busy}>
            {busy ? "Saving…" : "Agree and continue"}
          </button>
          <button className="gf-btn-ghost" onClick={onCancel}>Not now</button>
        </div>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Bits                                                               */
/* ------------------------------------------------------------------ */

const STATUS_COLOR = { proven: "#cdeb45", partial: "#f8e9cd", missing: "#fadfda" };

function MarkedJD({ jd, reqs, selected, onSelect }) {
  const spans = [];
  const used = [];
  reqs.forEach((r) => {
    if (!r.jd_quote) return;
    const idx = jd.toLowerCase().indexOf(r.jd_quote.toLowerCase().trim());
    if (idx === -1) return;
    const end = idx + r.jd_quote.trim().length;
    if (used.some((u) => idx < u[1] && end > u[0])) return;
    used.push([idx, end]);
    spans.push({ start: idx, end, req: r });
  });
  spans.sort((a, b) => a.start - b.start);

  const out = [];
  let cursor = 0;
  spans.forEach((s, i) => {
    if (s.start > cursor) out.push(<span key={"t" + i}>{jd.slice(cursor, s.start)}</span>);
    out.push(
      <button
        key={"m" + i}
        className={"gf-mk gf-mk-" + s.req.status}
        data-sel={selected === s.req.id}
        onClick={() => onSelect(selected === s.req.id ? null : s.req.id)}
        title={s.req.status === "missing" ? "Nothing in your resume proves this" : s.req.evidence || ""}
      >
        {jd.slice(s.start, s.end)}
      </button>
    );
    cursor = s.end;
  });
  out.push(<span key="tail">{jd.slice(cursor)}</span>);
  return <div className="gf-jd">{out}</div>;
}

function Section({ n, title, right, children }) {
  return (
    <section className="gf-sec">
      <div className="gf-sec-head">
        <span className="gf-eyebrow gf-mono">{n}</span>
        <h2>{title}</h2>
        {right ? <span className="gf-eyebrow">{right}</span> : null}
      </div>
      {children}
    </section>
  );
}

/* ------------------------------------------------------------------ */
/*  App                                                                */
/* ------------------------------------------------------------------ */

export default function GapFolio() {
  const [tab, setTab] = useState("analyze");
  const [resume, setResume] = useState("");
  const [jd, setJd] = useState("");
  const [stage, setStage] = useState(0); // 0 idle, 1 parsing, 2 matching, 3 fixes, 4 done
  const [match, setMatch] = useState(null);
  const [fixes, setFixes] = useState(null);
  const [err, setErr] = useState("");
  const [sel, setSel] = useState(null);
  const [signedIn, setSignedIn] = useState(!!api.getToken());
  const [gate, setGate] = useState(null); // null | "auth" | "consent"
  const fileRef = useRef(null);
  const resultRef = useRef(null);
  const resumeAfterGate = useRef(false);

  useEffect(() => {
    if (stage === 4 && resultRef.current) {
      resultRef.current.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }, [stage]);

  const loadSample = () => {
    setResume(SAMPLE_RESUME);
    setJd(SAMPLE_JD);
    setErr("");
    setMatch(null);
    setFixes(null);
    setStage(0);
  };

  const showSampleReport = () => {
    setMatch(SAMPLE_MATCH);
    setFixes(SAMPLE_FIXES);
    if (!jd) setJd(SAMPLE_JD);
    if (!resume) setResume(SAMPLE_RESUME);
    setErr("");
    setStage(4);
  };

  const onFile = (e) => {
    const f = e.target.files && e.target.files[0];
    if (!f) return;
    const r = new FileReader();
    r.onload = () => setResume(String(r.result || ""));
    r.readAsText(f);
  };

  const run = async () => {
    // Gate first: no token means the upload endpoint is unreachable anyway.
    if (!api.getToken()) {
      resumeAfterGate.current = true;
      setGate("auth");
      return;
    }

    setErr("");
    setMatch(null);
    setFixes(null);
    setSel(null);
    setStage(1);
    try {
      const doc = await api.pasteResume(resume);
      setStage(2);
      const report = await api.analyze(doc.id, jd);
      setStage(3);
      const vm = api.toViewModel(report);
      setMatch(vm.match);
      setFixes(vm.fixes);
      setStage(4);
    } catch (e) {
      setStage(0);
      if (e.status === 401) {
        // Token expired or revoked — drop it and re-gate.
        api.clearToken();
        setSignedIn(false);
        resumeAfterGate.current = true;
        setGate("auth");
        return;
      }
      if (api.isConsentBlock(e.status)) {
        // Consent missing or withdrawn since last time.
        resumeAfterGate.current = true;
        setGate("consent");
        return;
      }
      setErr(e.message || "Something broke on the way to the analysis.");
    }
  };

  const afterAuth = (res) => {
    setSignedIn(true);
    // Register returns the scopes still outstanding; login returns [] once granted.
    if ((res.consent_required || []).length) {
      setGate("consent");
      return;
    }
    setGate(null);
    if (resumeAfterGate.current) {
      resumeAfterGate.current = false;
      run();
    }
  };

  const afterConsent = () => {
    setGate(null);
    if (resumeAfterGate.current) {
      resumeAfterGate.current = false;
      run();
    }
  };

  const cancelGate = () => {
    resumeAfterGate.current = false;
    setGate(null);
  };

  const signOut = () => {
    api.clearToken();
    setSignedIn(false);
    setMatch(null);
    setFixes(null);
    setStage(0);
  };

  const busy = stage > 0 && stage < 4;
  const ready = resume.trim().length > 80 && jd.trim().length > 80;

  const counts = match
    ? match.requirements.reduce(
        (a, r) => ({ ...a, [r.status]: (a[r.status] || 0) + 1 }),
        { proven: 0, partial: 0, missing: 0 }
      )
    : null;

  return (
    <div className="gf">
      <style>{CSS}</style>

      {gate === "auth" && <AuthSheet onDone={afterAuth} onCancel={cancelGate} />}
      {gate === "consent" && <ConsentSheet onDone={afterConsent} onCancel={cancelGate} />}

      <div className="gf-shell">
        <header className="gf-top">
          <div className="gf-wordmark">
            <span className="hl">GAP</span>
            <span>FOLIO</span>
          </div>
          <nav className="gf-tabs">
            {[
              ["analyze", "Analyse a resume"],
              ["batch", "Batch dashboard"],
              ["how", "How it works"],
            ].map(([k, label]) => (
              <button key={k} className="gf-tab" data-on={tab === k} onClick={() => setTab(k)}>
                {label}
              </button>
            ))}
          </nav>
          <div className="gf-acct">
            {signedIn ? (
              <>
                <span>signed in</span>
                <button className="gf-btn-ghost" onClick={signOut}>Sign out</button>
              </>
            ) : (
              <button className="gf-btn-ghost" onClick={() => setGate("auth")}>Sign in</button>
            )}
          </div>
        </header>

        {tab === "analyze" && (
          <>
            {!match && !busy && (
              <div className="gf-hero">
                <div className="gf-eyebrow">Resume → job description gap analysis</div>
                <h1>
                  Know exactly why <span className="strike">you didn't</span> get the interview.
                </h1>
                <p>
                  A recruiter gives your resume six seconds and sends back nothing. GapFolio holds your
                  resume against one job description, marks up every line the JD asks for, and hands you the
                  fix for each gap — a project, a course, or a rewritten bullet.
                </p>
                <div className="gf-legend">
                  <span>
                    <i className="gf-swatch" style={{ background: "var(--marker)" }} /> proven
                  </span>
                  <span>
                    <i className="gf-swatch" style={{ background: "var(--amber-soft)", borderBottom: "2px solid var(--amber)" }} /> claimed, not proven
                  </span>
                  <span>
                    <i className="gf-swatch" style={{ background: "var(--redline-soft)" }} /> missing
                  </span>
                </div>
              </div>
            )}

            <div className="gf-grid">
              <div className="gf-card">
                <div className="gf-card-head">
                  <h3>Your resume</h3>
                  <button className="gf-btn-ghost" onClick={() => fileRef.current && fileRef.current.click()}>
                    Upload .txt
                  </button>
                  <input ref={fileRef} type="file" accept=".txt,.md" onChange={onFile} style={{ display: "none" }} />
                </div>
                <textarea
                  className="gf-ta"
                  value={resume}
                  onChange={(e) => setResume(e.target.value)}
                  placeholder={"Paste your resume text here.\n\nFrom a PDF: open it, select all, copy, paste. Parsing PDFs and .docx server-side is on the build list."}
                />
              </div>

              <div className="gf-card">
                <div className="gf-card-head">
                  <h3>Target job description</h3>
                  <span className="gf-eyebrow">one role at a time</span>
                </div>
                <textarea
                  className="gf-ta"
                  value={jd}
                  onChange={(e) => setJd(e.target.value)}
                  placeholder={"Paste the full job posting — responsibilities and requirements both.\n\nThe more of the posting you paste, the more precise the markup."}
                />
              </div>
            </div>

            <div className="gf-actions">
              <button className="gf-btn" onClick={run} disabled={!ready || busy}>
                {busy ? "Analysing…" : "Run the gap analysis"}
              </button>
              <button className="gf-btn-ghost" onClick={loadSample}>
                Load a sample resume + JD
              </button>
              <button className="gf-btn-ghost" onClick={showSampleReport}>
                See a finished report
              </button>
              {!ready && !busy && <span className="gf-note">Both boxes need real content before this can run.</span>}
            </div>

            {err && (
              <div className="gf-err">
                <strong>{err}</strong>
                <div style={{ marginTop: 8 }}>
                  Check your connection and run it again, or{" "}
                  <button className="gf-btn-ghost" onClick={showSampleReport} style={{ marginLeft: 4 }}>
                    open the sample report
                  </button>{" "}
                  — the backup for a live demo.
                </div>
              </div>
            )}

            {busy && (
              <div className="gf-scan">
                <div className="gf-scan-line" />
                <div className="gf-eyebrow" style={{ color: "#7f8a96" }}>
                  Six seconds is all a recruiter gives it. This takes a little longer.
                </div>
                <div className="gf-stages">
                  {[
                    [1, "Parsing and chunking your resume"],
                    [2, "Matching on meaning, not keywords"],
                    [3, "Writing the fix for every gap"],
                  ].map(([n, label]) => (
                    <div key={n} className="gf-stage" data-on={stage === n} data-done={stage > n}>
                      <i className="gf-dot" />
                      {label}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {match && (
              <div ref={resultRef}>
                {match.parse_confidence?.level === "low" && (
                  <div
                    role="alert"
                    style={{
                      border: "1px solid var(--amber)",
                      background: "#fff8e6",
                      padding: "12px 16px",
                      marginBottom: 20,
                      fontSize: 14,
                    }}
                  >
                    <strong>
                      We could only read part of this resume ({Math.round((match.parse_confidence.score || 0) * 100)}%
                      parse confidence).
                    </strong>{" "}
                    Treat the gaps below as provisional until the file is fixed.
                    {(match.parse_confidence.notes || []).length > 0 && (
                      <ul style={{ margin: "6px 0 0 18px" }}>
                        {match.parse_confidence.notes.map((n, i) => (
                          <li key={i}>{n}</li>
                        ))}
                      </ul>
                    )}
                  </div>
                )}
                <Section n="01" title="Where you stand" right={match.role_title}>
                  <div className="gf-score">
                    <div>
                      <div className="gf-score-num">
                        {match.match_score}
                        <sup>/100</sup>
                      </div>
                      <div className="gf-eyebrow" style={{ marginTop: 12, color: "#7f8a96" }}>
                        match score
                      </div>
                    </div>
                    <div>
                      <h3 className="gf-verdict">
                        {match.readiness === "Ready"
                          ? "Ready to apply"
                          : match.readiness === "Close"
                          ? "Close — two fixes away"
                          : "Not yet — but the path is short"}
                      </h3>
                      <p>{match.headline}</p>
                      <div className="gf-coverage">
                        {match.requirements.map((r) => (
                          <i key={r.id} style={{ background: STATUS_COLOR[r.status] || "#333" }} title={r.label} />
                        ))}
                      </div>
                      <div className="gf-cov-key">
                        <span>{counts.proven} proven</span>
                        <span>{counts.partial} unproven</span>
                        <span>{counts.missing} missing</span>
                      </div>
                    </div>
                  </div>
                </Section>

                <Section n="02" title="The job description, marked up" right="click any mark">
                  <div className="gf-two">
                    <MarkedJD jd={jd} reqs={match.requirements} selected={sel} onSelect={setSel} />
                    <div className="gf-card">
                      <h3 style={{ marginBottom: 12 }}>Requirement ledger</h3>
                      {match.requirements.map((r, i) => (
                        <div
                          key={r.id}
                          className="gf-row"
                          style={{ background: sel === r.id ? "#f7f9f2" : "transparent", cursor: "pointer" }}
                          onClick={() => setSel(sel === r.id ? null : r.id)}
                        >
                          <div className="gf-idx">{String(i + 1).padStart(2, "0")}</div>
                          <div>
                            <div className="gf-req">{r.label}</div>
                            <div className="gf-ev">
                              {r.status === "missing" ? "Nothing on the page supports this." : r.evidence}
                            </div>
                          </div>
                          <div className={"gf-pill gf-pill-" + r.status}>{r.status}</div>
                        </div>
                      ))}
                    </div>
                  </div>
                </Section>

                {fixes && fixes.roadmap.length > 0 && (
                  <Section n="03" title="Your gap roadmap" right="highest impact first">
                    <div className="gf-stack">
                      {fixes.roadmap.map((f, i) => (
                        <div key={i} className="gf-fix" data-p={f.priority}>
                          <div className="gf-fix-top">
                            <h4>{f.skill}</h4>
                            <span className="gf-eyebrow">
                              {f.fix_type} · {f.effort} · {f.priority} priority
                            </span>
                          </div>
                          <p>{f.action}</p>
                          {f.proof && (
                            <div className="gf-proof">
                              <span style={{ opacity: 0.55 }}>the bullet you earn — </span>
                              {f.proof}
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  </Section>
                )}

                {fixes && fixes.rewrites.length > 0 && (
                  <Section n="04" title="Rewrite these bullets" right="before → after">
                    <div className="gf-stack">
                      {fixes.rewrites.map((r, i) => (
                        <div key={i} className="gf-diff">
                          <div className="old">{r.before}</div>
                          <div className="new">{r.after}</div>
                          <div className="why">{r.why}</div>
                        </div>
                      ))}
                    </div>
                    <p className="gf-note" style={{ marginTop: 12 }}>
                      Numbers in the rewrites are placeholders shaped like the real thing. Replace them with yours
                      before this goes anywhere near an application.
                    </p>
                  </Section>
                )}

                {fixes && fixes.ats.length > 0 && (
                  <Section n="05" title="Will a machine read it?" right="ATS parsing check">
                    <div className="gf-two">
                      <div className="gf-card">
                        {fixes.ats.map((c, i) => (
                          <div key={i}>
                            {(i === 0 || fixes.ats[i - 1].group !== c.group) && c.group && (
                              <div className="gf-eyebrow" style={{ margin: "12px 0 6px" }}>
                                {c.group === "parse" ? "Can a machine read the file?" : "Does it read well once parsed?"}
                              </div>
                            )}
                          <div className="gf-check">
                            <span
                              className="gf-chip"
                              style={{
                                background:
                                  c.status === "pass" ? "#7fa310" : c.status === "warn" ? "var(--amber)" : "var(--redline)",
                              }}
                            >
                              {c.status === "pass" ? "✓" : c.status === "warn" ? "!" : "✗"}
                            </span>
                            <div>
                              <div style={{ fontWeight: 600, fontSize: 14 }}>{c.check}</div>
                              <div className="gf-ev">{c.note}</div>
                            </div>
                          </div>
                          </div>
                        ))}
                      </div>
                      {fixes.questions.length > 0 && (
                        <div className="gf-card">
                          <h3 style={{ marginBottom: 10 }}>Questions this resume invites</h3>
                          <p className="gf-note" style={{ marginTop: 0, marginBottom: 14 }}>
                            Generated from the gaps above. If you can't answer these, the gap is still open.
                          </p>
                          {fixes.questions.map((q, i) => (
                            <div key={i} className="gf-check">
                              <span className="gf-idx gf-mono">{String(i + 1).padStart(2, "0")}</span>
                              <div style={{ fontSize: 13.5 }}>{q}</div>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  </Section>
                )}

                <div className="gf-actions">
                  <button
                    className="gf-btn"
                    onClick={() => {
                      setMatch(null);
                      setFixes(null);
                      setStage(0);
                      window.scrollTo({ top: 0, behavior: "smooth" });
                    }}
                  >
                    Analyse another role
                  </button>
                  <span className="gf-note">
                    In the full build this report is saved, so progress across applications is tracked over time.
                  </span>
                </div>
              </div>
            )}
          </>
        )}

        {tab === "batch" && (
          <>
            <div className="gf-hero">
              <div className="gf-eyebrow">B2B · placement cell view</div>
              <h1 style={{ fontSize: "clamp(32px,5vw,52px)" }}>See the batch before the rejections do.</h1>
              <p>
                Every analysis a student runs rolls up here. Placement officers see readiness across the graduating
                class in one screen and can reach the bottom quartile in week one instead of month six.
              </p>
              <p className="gf-note" style={{ marginTop: 14 }}>
                Sample data — 294 students, VIT Pune, 2027 batch.
              </p>
            </div>

            <div className="gf-kpis">
              {[
                ["Resumes analysed", "294", "of 341 in batch"],
                ["Median match score", "63", "target roles, Sep cycle"],
                ["At risk (below 50)", "119", "40% of the batch"],
                ["Most common gap", "Docker / CI", "in 71% of reports"],
              ].map(([label, val, sub]) => (
                <div key={label} className="gf-kpi">
                  <div className="gf-eyebrow">{label}</div>
                  <b>{val}</b>
                  <div className="gf-note" style={{ marginTop: 4 }}>{sub}</div>
                </div>
              ))}
            </div>

            <Section n="01" title="Readiness distribution" right="match score bands">
              <div className="gf-card" style={{ height: 260 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={DIST} margin={{ top: 10, right: 8, left: -18, bottom: 0 }}>
                    <XAxis
                      dataKey="band"
                      tick={{ fontSize: 11, fontFamily: "IBM Plex Mono", fill: "#5c6672" }}
                      axisLine={{ stroke: "#d7dbd1" }}
                      tickLine={false}
                    />
                    <YAxis
                      tick={{ fontSize: 11, fontFamily: "IBM Plex Mono", fill: "#5c6672" }}
                      axisLine={false}
                      tickLine={false}
                    />
                    <Tooltip
                      cursor={{ fill: "#f2f4ee" }}
                      contentStyle={{
                        border: "1px solid #d7dbd1",
                        borderRadius: 0,
                        fontFamily: "IBM Plex Mono",
                        fontSize: 12,
                      }}
                      formatter={(v) => [v + " students", "count"]}
                    />
                    <Bar dataKey="n">
                      {DIST.map((d, i) => (
                        <Cell key={i} fill={d.color} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </Section>

            <Section n="02" title="Students to reach first" right="sorted by score">
              <table className="gf-table">
                <thead>
                  <tr>
                    <th>Student</th>
                    <th>Branch</th>
                    <th>Match score</th>
                    <th>Top gap</th>
                    <th>Applications</th>
                  </tr>
                </thead>
                <tbody>
                  {[...BATCH]
                    .sort((a, b) => a.score - b.score)
                    .map((s) => (
                      <tr key={s.name}>
                        <td style={{ fontWeight: 600 }}>{s.name}</td>
                        <td className="gf-mono" style={{ fontSize: 12, color: "var(--graphite)" }}>{s.branch}</td>
                        <td>
                          <span className="gf-bar">
                            <i
                              style={{
                                width: s.score + "%",
                                background: s.score < 50 ? "var(--redline)" : s.score < 70 ? "var(--amber)" : "#7fa310",
                              }}
                            />
                          </span>
                          <span className="gf-mono" style={{ fontSize: 12 }}>{s.score}</span>
                        </td>
                        <td style={{ color: "var(--graphite)" }}>{s.gap}</td>
                        <td className="gf-mono" style={{ fontSize: 12 }}>{s.apps}</td>
                      </tr>
                    ))}
                </tbody>
              </table>
              <p className="gf-note" style={{ marginTop: 12 }}>
                Aditya Rao's resume fails ATS parsing entirely — one afternoon of formatting work moves him further
                than another month of applications.
              </p>
            </Section>
          </>
        )}

        {tab === "how" && (
          <>
            <div className="gf-hero">
              <div className="gf-eyebrow">Under the hood</div>
              <h1 style={{ fontSize: "clamp(32px,5vw,52px)" }}>Semantic matching, then a fix for every gap.</h1>
              <p>
                Keyword matchers reject "client engagement" when the posting says "customer relationship management".
                GapFolio compares meaning, then does the part nobody else does: it says what to do next.
              </p>
            </div>

            <Section n="01" title="The pipeline" right="five steps">
              <div className="gf-stack">
                {[
                  ["Extract", "An LLM pulls structured entities out of both documents — skills, seniority, responsibilities, tools — instead of splitting on whitespace."],
                  ["Match", "Sentence embeddings score each JD requirement against the resume, so equivalent phrasing counts as a match and a bare skills-list mention doesn't count as proof."],
                  ["Flag", "A rule-based checker runs over formatting: parseable headings, quantified outcomes, weak verbs, missing links — the things that kill a resume before a human sees it."],
                  ["Prescribe", "Every gap is mapped to one concrete action and the resume bullet it would earn. This is the part existing tools stop short of."],
                  ["Learn", "Reported outcomes — interview or no interview — feed back into scoring, so the model improves on real callbacks rather than on assumptions."],
                ].map(([t, d], i) => (
                  <div key={t} className="gf-fix" data-p="medium" style={{ borderLeftColor: "var(--ink)" }}>
                    <div className="gf-fix-top">
                      <h4>{t}</h4>
                      <span className="gf-eyebrow">step {String(i + 1).padStart(2, "0")}</span>
                    </div>
                    <p>{d}</p>
                  </div>
                ))}
              </div>
            </Section>

            <Section n="02" title="What this prototype does and doesn't do">
              <div className="gf-two">
                <div className="gf-card">
                  <h3 style={{ marginBottom: 12 }}>Working now</h3>
                  {[
                    "Server-side extraction and matching against any pasted JD",
                    "Marked-up job description with per-requirement evidence",
                    "Generated roadmap, bullet rewrites and ATS checks",
                    "Accounts, granular consent and saved reports",
                    "Sample batch dashboard for the placement-cell pitch",
                  ].map((x) => (
                    <div key={x} className="gf-check">
                      <span className="gf-chip" style={{ background: "#7fa310" }}>✓</span>
                      <div style={{ fontSize: 13.5 }}>{x}</div>
                    </div>
                  ))}
                </div>
                <div className="gf-card">
                  <h3 style={{ marginBottom: 12 }}>Still to build</h3>
                  {[
                    "PDF and .docx upload from this screen (the API parses both already)",
                    "Progress tracking across applications, surfaced in the UI",
                    "Outcome capture — the feedback loop that makes the score predictive",
                    "Batch dashboard wired to real placement-cell data",
                    "PII redaction before resume text reaches third-party models",
                  ].map((x) => (
                    <div key={x} className="gf-check">
                      <span className="gf-chip" style={{ background: "var(--amber)" }}>!</span>
                      <div style={{ fontSize: 13.5 }}>{x}</div>
                    </div>
                  ))}
                </div>
              </div>
            </Section>
          </>
        )}

        <footer className="gf-foot">
          <span>GapFolio — prototype built from the project documentation and pitch deck.</span>
          <span className="gf-mono">
            Resumes are stored against your account and deleted 365 days after their last analysis.
            You can withdraw consent or delete everything at any time.
          </span>
        </footer>
      </div>
    </div>
  );
}
