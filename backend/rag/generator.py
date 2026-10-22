"""
LLM generator — calls local Ollama to produce a root-cause diagnosis.

The prompt contains the failing log, similar past failures and (only when strong enough) a known
issue. Evidence is marked as possibly unrelated so the model uses it only when it clearly fits.
If Ollama is unavailable or returns invalid output, the diagnosis falls back to the KB entry.

call_llm() and _build_prompt(ask_category=True) are also used by evaluation/run_rag_ablation.py.
"""

import json
from dataclasses import dataclass
from typing import Optional

import httpx
from pydantic import BaseModel, Field, ValidationError, field_validator

from ingestion.log_parser import ParsedLog
from rag.confidence import confidence_from_similarity
from rag.retriever import RetrievedEvidence

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"   # change to "mistral", "llama3", "qwen2.5", ...
FALLBACK_MODE_MSG = "(KB fallback — Ollama unavailable)"
FALLBACK_BAD_OUTPUT_MSG = "(KB fallback — invalid LLM output)"
TIMEOUT = 120  # seconds
LOG_EXCERPT_CHARS = 1500

CATEGORIES = ["assertion_failure", "dependency_or_import", "build_or_compile",
              "runtime_exception", "environment_or_infra", "other"]
CATEGORY_LEGEND = """Categories:
- assertion_failure: a test compared an expected value with an actual value and they differed
- dependency_or_import: a package, module, class or library version is missing or conflicting
- build_or_compile: the code failed to compile, type-check or build
- runtime_exception: the code raised an unexpected exception (null, type, value, index errors)
- environment_or_infra: timeout, network, file path, memory or disk, configuration, flaky or race problems, CI runner issues
- other: none of the above"""


@dataclass
class DiagnosisResult:
    root_cause: str
    confidence: float           # calibrated retrieval confidence (rag/confidence.py)
    explanation: str
    next_steps: list[str]
    fix_snippet: Optional[str]
    model_used: str
    similar_failures_used: int
    kb_entry_used: Optional[str]  # KB entry ID if one was given to the model


class LLMDiagnosis(BaseModel):
    """Schema the LLM's JSON must satisfy; anything else triggers the KB fallback."""
    root_cause: str = Field(min_length=1)
    explanation: Optional[str] = ""
    next_steps: list[str] = Field(default_factory=list)
    fix_snippet: Optional[str] = None
    category: Optional[str] = None  # only requested by the evaluation prompt

    @field_validator("next_steps", mode="before")
    @classmethod
    def _coerce_steps(cls, v):
        if isinstance(v, str):
            return [v]
        if isinstance(v, list):
            return [str(s) for s in v]
        return []

    @field_validator("fix_snippet", mode="before")
    @classmethod
    def _coerce_snippet(cls, v):
        if v is None or isinstance(v, str):
            return v or None
        return str(v)


def _build_prompt(parsed_log: ParsedLog, evidence: RetrievedEvidence, ask_category: bool = False) -> str:
    similar_ctx = ""
    for i, (rec, score) in enumerate(evidence.similar_failures[:3], 1):
        similar_ctx += (
            f"\n[Past failure {i}] (similarity {score:.2f})\n"
            f"  Test: {rec.test_name}\n"
            f"  Error: {rec.error_type} — {rec.error_message}\n"
            f"  Module: {rec.module}\n"
        )
        if rec.resolution:
            similar_ctx += f"  Resolution: {rec.resolution}\n"

    kb_ctx = ""
    if evidence.top_kb_entry:
        kb = evidence.top_kb_entry
        kb_ctx = (
            f"\n[{kb.id}] {kb.title}\n"
            f"  Typical root cause: {kb.root_cause}\n"
            f"  Typical fix: {kb.resolution}\n"
            f"  Example: {kb.fix_snippet}\n"
        )

    category_block = f"\n{CATEGORY_LEGEND}\n" if ask_category else ""
    category_field = (
        f'  "category": "exactly one of: {", ".join(CATEGORIES)}",\n' if ask_category else ""
    )
    excerpt = parsed_log.raw_log[:LOG_EXCERPT_CHARS]
    return f"""You are an expert at diagnosing failing CI tests.

=== FAILING TEST ===
Framework: {parsed_log.framework}
Test: {parsed_log.test_name}
Module: {parsed_log.module}
Error: {parsed_log.error_type}: {parsed_log.error_message}

=== LOG EXCERPT (truncated) ===
{excerpt}

=== SIMILAR PAST FAILURES (retrieved automatically, may be unrelated) ==={similar_ctx if similar_ctx else " (none)"}

=== POSSIBLY RELATED KNOWN ISSUE (retrieved automatically, often wrong) ==={kb_ctx if kb_ctx else " (none)"}

=== INSTRUCTIONS ===
Diagnose THIS failure from the log. Use a past failure or the known issue only if it clearly matches the
error above; if it does not match, ignore it. Do not invent details that are not in the log.
{category_block}Return valid JSON with exactly this schema:
{{
  "root_cause": "1-2 sentences",
{category_field}  "explanation": "2-4 sentences",
  "next_steps": ["step 1", "step 2", "step 3"],
  "fix_snippet": "short code or command, or an empty string"
}}
Only output the JSON object."""


def call_llm(prompt: str, num_predict: int = 800, temperature: float = 0.2, seed: Optional[int] = None) -> dict:
    """Call Ollama and return the validated diagnosis as a dict. Raises on any failure."""
    options = {"temperature": temperature, "num_predict": num_predict}
    if seed is not None:
        options["seed"] = seed
    response = httpx.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "keep_alive": "30m",
            "options": options,
        },
        timeout=TIMEOUT,
    )
    response.raise_for_status()
    return LLMDiagnosis.model_validate(json.loads(response.json()["response"])).model_dump()


def _confidence(evidence: RetrievedEvidence) -> float:
    return confidence_from_similarity(evidence.top_similarity)


def generate_diagnosis(parsed_log: ParsedLog, evidence: RetrievedEvidence) -> DiagnosisResult:
    """Ask Ollama for a diagnosis; fall back to KB data if anything goes wrong."""
    prompt = _build_prompt(parsed_log, evidence)
    model_used = OLLAMA_MODEL

    try:
        data = call_llm(prompt)
    except httpx.HTTPError as e:  # connection refused, timeout, 404/500 from Ollama
        print(f"[Generator] Ollama request failed: {type(e).__name__}: {e}")
        data = _fallback_from_evidence(evidence)
        model_used = FALLBACK_MODE_MSG
    except (json.JSONDecodeError, KeyError, ValidationError) as e:
        print(f"[Generator] Invalid LLM output: {type(e).__name__}: {e}")
        data = _fallback_from_evidence(evidence)
        model_used = FALLBACK_BAD_OUTPUT_MSG

    return DiagnosisResult(
        root_cause=data.get("root_cause") or "Unable to determine root cause.",
        confidence=_confidence(evidence),
        explanation=data.get("explanation") or "",
        next_steps=data.get("next_steps", []),
        fix_snippet=data.get("fix_snippet"),
        model_used=model_used,
        similar_failures_used=len(evidence.similar_failures),
        kb_entry_used=evidence.top_kb_entry.id if evidence.top_kb_entry else None,
    )


def _fallback_from_evidence(evidence: RetrievedEvidence) -> dict:
    """Structured fallback when Ollama is unavailable: uses KB data directly."""
    if evidence.top_kb_entry:
        kb = evidence.top_kb_entry
        return {
            "root_cause": kb.root_cause,
            "explanation": kb.description,
            "next_steps": [kb.resolution],
            "fix_snippet": kb.fix_snippet,
        }
    return {
        "root_cause": "No similar failures or known issues found.",
        "explanation": "The retrieval system found no matching evidence for this failure.",
        "next_steps": [
            "Review the stacktrace manually.",
            "Check recent code changes in the affected module.",
            "Search the codebase for the failing method.",
        ],
        "fix_snippet": None,
    }