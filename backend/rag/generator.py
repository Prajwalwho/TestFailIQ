"""
LLM generator — calls local Ollama to produce root-cause diagnosis.
Uses a structured RAG prompt with retrieved evidence as context.
"""

import httpx
import json
from dataclasses import dataclass
from typing import Optional

from ingestion.log_parser import ParsedLog
from rag.retriever import RetrievedEvidence

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"   # Change to "mistral", "llama3", "codellama", etc.
FALLBACK_MODE_MSG = "(KB fallback — Ollama not running)"
TIMEOUT = 120  # seconds


@dataclass
class DiagnosisResult:
    root_cause: str
    confidence: float           # 0.0–1.0, derived from top retrieval score
    explanation: str
    next_steps: list[str]
    fix_snippet: Optional[str]
    model_used: str
    similar_failures_used: int
    kb_entry_used: Optional[str]  # KB entry ID if matched


def _build_prompt(parsed_log: ParsedLog, evidence: RetrievedEvidence) -> str:
    """Construct the RAG prompt injecting retrieved evidence as few-shot context."""

    # --- Historical similar failures (up to 3) ---
    similar_ctx = ""
    for i, (rec, score) in enumerate(evidence.similar_failures[:3], 1):
        similar_ctx += (
            f"\n[Similar Failure {i}] (similarity: {score:.2f})\n"
            f"  Test: {rec.test_name}\n"
            f"  Error: {rec.error_type} — {rec.error_message}\n"
            f"  Module: {rec.module}\n"
        )
        if rec.resolution:
            similar_ctx += f"  Resolution: {rec.resolution}\n"

    # --- KB evidence ---
    kb_ctx = ""
    if evidence.top_kb_entry:
        kb = evidence.top_kb_entry
        kb_ctx = (
            f"\n[Known Issue: {kb.id}] {kb.title}\n"
            f"  Root cause: {kb.root_cause}\n"
            f"  Resolution: {kb.resolution}\n"
            f"  Fix: {kb.fix_snippet}\n"
        )

    prompt = f"""You are an expert software test failure analyst. 
Your job is to analyze a failing test log, and using the retrieved historical failures and known issues below, produce a concise root-cause diagnosis and actionable next steps.

=== CURRENT FAILING TEST ===
Test: {parsed_log.test_name}
Module: {parsed_log.module}
Error Type: {parsed_log.error_type}
Error Message: {parsed_log.error_message}
Stacktrace:
{parsed_log.stacktrace}

=== RETRIEVED SIMILAR HISTORICAL FAILURES ==={similar_ctx if similar_ctx else " (none found)"}

=== MATCHING KNOWN ISSUES ==={kb_ctx if kb_ctx else " (none found)"}

=== YOUR TASK ===
Based on the above evidence, provide:
1. Root Cause: A precise 1-2 sentence explanation of WHY this test is failing.
2. Explanation: A 2-4 sentence deeper technical explanation.
3. Next Steps: 3-5 concrete, actionable steps to fix this (numbered list).
4. Fix Snippet: If applicable, a short code fix.

Respond in valid JSON with this exact schema:
{{
  "root_cause": "...",
  "explanation": "...",
  "next_steps": ["step 1", "step 2", "step 3"],
  "fix_snippet": "..." 
}}
Only output the JSON object, nothing else."""

    return prompt


def generate_diagnosis(parsed_log: ParsedLog, evidence: RetrievedEvidence) -> DiagnosisResult:
    """Call Ollama to generate a root-cause diagnosis using the RAG prompt."""
    prompt = _build_prompt(parsed_log, evidence)

    try:
        response = httpx.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "format": "json",
            },
            timeout=TIMEOUT,
        )
        response.raise_for_status()
        raw = response.json()["response"]
        data = json.loads(raw)
    except httpx.ConnectError:
        # Ollama not running — return a structured fallback
        data = _fallback_from_evidence(evidence)
        return DiagnosisResult(
            root_cause=data.get("root_cause", "Unable to determine root cause."),
            confidence=round(min(evidence.top_similarity + 0.2, 1.0) if evidence.top_kb_entry else evidence.top_similarity, 3),
            explanation=data.get("explanation", ""),
            next_steps=data.get("next_steps", []),
            fix_snippet=data.get("fix_snippet"),
            model_used=FALLBACK_MODE_MSG,
            similar_failures_used=len(evidence.similar_failures),
            kb_entry_used=evidence.top_kb_entry.id if evidence.top_kb_entry else None,
        )
    except (json.JSONDecodeError, KeyError):
        data = _fallback_from_evidence(evidence)

    # Confidence = top retrieval similarity, boosted if KB matched
    confidence = evidence.top_similarity
    if evidence.top_kb_entry and confidence < 0.5:
        confidence = min(confidence + 0.2, 1.0)

    return DiagnosisResult(
        root_cause=data.get("root_cause", "Unable to determine root cause."),
        confidence=round(confidence, 3),
        explanation=data.get("explanation", ""),
        next_steps=data.get("next_steps", []),
        fix_snippet=data.get("fix_snippet"),
        model_used=OLLAMA_MODEL,
        similar_failures_used=len(evidence.similar_failures),
        kb_entry_used=evidence.top_kb_entry.id if evidence.top_kb_entry else None,
    )


def _fallback_from_evidence(evidence: RetrievedEvidence) -> dict:
    """Structured fallback when Ollama is unavailable — uses KB data directly."""
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
