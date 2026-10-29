/* ------------------------------------------------------------------ */
/*  GapFolio API client                                                */
/*                                                                     */
/*  Talks to the FastAPI backend. Every resume route needs a bearer    */
/*  token AND a granted `resume_processing` consent — consent is a     */
/*  route dependency server-side, so a missing scope is a 403, not a   */
/*  degraded result.                                                   */
/* ------------------------------------------------------------------ */

// In development the UI (Vite, :5173) and API (:8000) are separate origins. In a
// production build the API serves the UI itself, so requests are same-origin.
const BASE = import.meta.env.VITE_API_BASE ?? (import.meta.env.DEV ? "http://localhost:8000" : "");
const TOKEN_KEY = "gf_token";

export class ApiError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

export const getToken = () => localStorage.getItem(TOKEN_KEY) || "";
export const setToken = (t) => localStorage.setItem(TOKEN_KEY, t);
export const clearToken = () => localStorage.removeItem(TOKEN_KEY);

/** Pull a readable message out of FastAPI's error shapes. */
function readError(status, body) {
  const d = body && body.detail;
  if (typeof d === "string") return d;
  if (Array.isArray(d) && d.length) return d[0].msg || "Invalid request.";
  if (status === 401) return "Your session expired. Sign in again.";
  if (status === 403 || status === 451) return "Consent is required before this can run.";
  return `The server returned ${status}.`;
}

/** The backend answers a missing/withdrawn consent scope with 451, not 403. */
export const isConsentBlock = (status) => status === 451 || status === 403;

async function call(path, { method = "GET", body, auth = true } = {}) {
  const headers = {};
  if (body !== undefined) headers["Content-Type"] = "application/json";
  if (auth) {
    const t = getToken();
    if (t) headers["Authorization"] = `Bearer ${t}`;
  }

  let res;
  try {
    res = await fetch(BASE + path, {
      method,
      headers,
      body: body === undefined ? undefined : JSON.stringify(body),
    });
  } catch {
    throw new ApiError(0, `Can't reach the analysis server at ${BASE}. Is it running?`);
  }

  const text = await res.text();
  let data = null;
  if (text) {
    try {
      data = JSON.parse(text);
    } catch {
      /* non-JSON body — fall through to the status-based message */
    }
  }

  if (!res.ok) throw new ApiError(res.status, readError(res.status, data));
  return data;
}

/* --- auth ---------------------------------------------------------- */

export const register = (email, password, full_name) =>
  call("/api/auth/register", { method: "POST", auth: false, body: { email, password, full_name } });

export const login = (email, password) =>
  call("/api/auth/login", { method: "POST", auth: false, body: { email, password } });

/* --- privacy ------------------------------------------------------- */

export const getPolicy = () => call("/api/privacy/policy", { auth: false });

export const setConsent = (scope, granted) =>
  call("/api/privacy/consents", { method: "POST", body: { scope, granted } });

/* --- analysis ------------------------------------------------------ */

export const pasteResume = (text) =>
  call("/api/resumes/paste", { method: "POST", body: { text } });

export const analyze = (resume_id, jd_text) =>
  call("/api/analyze", { method: "POST", body: { resume_id, jd_text } });

/**
 * Split one backend report into the `match` / `fixes` shapes the UI renders.
 * The backend returns both in a single response; it also omits `id` on
 * requirements, which the JD markup, ledger and coverage bar all key off.
 */
export function toViewModel(report) {
  const match = {
    role_title: report.role_title,
    seniority: report.seniority,
    match_score: Math.round(report.match_score ?? 0),
    readiness: report.readiness,
    headline: report.headline,
    parse_confidence: report.parse_confidence,
    skills_detected: report.skills_detected || [],
    requirements: (report.requirements || []).map((r, i) => ({ ...r, id: r.id ?? i + 1 })),
  };
  const fixes = {
    roadmap: report.roadmap || [],
    ats: report.ats || [],
    rewrites: report.rewrites || [],
    questions: report.questions || [],
  };
  return { match, fixes };
}
