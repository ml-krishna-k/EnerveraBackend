"""
Layered composition for the clinical answer system prompt.

The previous code embedded the entire prompt as one ~70-line f-string in two
places — `app.services.orchestration.pipeline._compose_answer_prompts` (the
FastAPI path) and `graphrag.llm.gemini_llm.GeminiLLM.generate_response` (the
legacy sync CLI). Both call sites now import `compose_system_prompt(...)`
from this module instead.

Token budget: the composed prompt is meant to fit under SYSTEM_PROMPT_MAX_TOKENS
(400 tokens / ~1600 chars). Layer content is therefore compressed — every
rule from the user's reference consultation prompt is preserved, but
elaboration / redundant phrasings / decorative examples are trimmed.

Layers (in compose order):
    1. core_identity            — gastroenterology consultation persona
    2. safety_policy            — emergency-number restraint, named red-
                                   flags vs named routine complaints,
                                   doctor-confirmation disclaimer
    3. runtime_modifiers        — risk header + name-known personalisation
    4. session_state            — STEP 0: prior conversation first, but
                                   answer the current question
    5. retrieval_grounding      — use snippets without leaking meta
    6. tool_instructions        — empty today; placeholder
    7. formatting_constraints   — STEP 1/2/3 one-question-at-a-time
                                   consultation flow for substantive turns
"""

from __future__ import annotations


# Risk tone surfaced at the top of the runtime layer. Keys: none | low |
# medium | high | critical (from analysis.risk_level). Low / none stay
# empty so the prompt doesn't carry irrelevant warnings.
_RISK_TONE: dict[str, str] = {
    "critical": (
        "⚠️ CRITICAL RISK — if signs are genuinely life-threatening "
        "(see safety section), SKIP the interview flow and escalate first."
    ),
    "high": (
        "⚠️ Elevated risk — be explicit about red flags and when to seek "
        "care; don't hedge urgency."
    ),
    "medium": "Note: moderate risk signals — be thorough and safety-aware.",
    "low": "",
    "none": "",
}


# Query types that warrant the full consultation flow. Anything else gets
# the short non-substantive treatment.
_SUBSTANTIVE_QUERY_TYPES: frozenset[str] = frozenset({
    "symptom_query", "diagnosis_query", "diagnosis",
    "medication_query", "treatment_query", "drug_interaction",
    "guideline", "lab_interpretation", "prognosis", "unknown",
})


# ---------------------------------------------------------------------------
# Layer 1 — Core identity
# ---------------------------------------------------------------------------

def layer_core_identity() -> str:
    return (
        "You are a warm, empathetic gastroenterology assistant — a "
        "knowledgeable friend with medical training. Consult like a real "
        "doctor: plain English, no jargon. Make the patient feel heard "
        "first, then helped."
    )


# ---------------------------------------------------------------------------
# Layer 2 — Safety policy
# ---------------------------------------------------------------------------

def layer_safety_policy() -> str:
    return (
        "SAFETY\n"
        "- Probabilistic language only; never give a definitive diagnosis. "
        "Include \"only a doctor can properly examine and confirm this\" "
        "in substantive replies.\n"
        "- Emergency numbers (112/102/108) ONLY for life-threatening signs: "
        "vomiting blood, severe chest pain, can't breathe, collapse, stroke "
        "signs, anaphylaxis, black tarry stool.\n"
        "- NEVER show emergency numbers for routine GI complaints "
        "(bloating, acidity, burping, constipation, mild cramps)."
    )


# ---------------------------------------------------------------------------
# Layer 3 — Runtime modifiers
# ---------------------------------------------------------------------------

def layer_runtime_modifiers(*, risk_level: str, has_name: bool) -> str:
    parts: list[str] = []

    risk_block = _RISK_TONE.get((risk_level or "none").lower(), "")
    if risk_block:
        parts.append(risk_block)

    if has_name:
        parts.append(
            "PERSONALISATION\n"
            "- Memory carries \"Patient name: <Name>\". Greet by that name "
            "on the first line of your first substantive reply (\"Hey "
            "Aarav,\"); then use sparingly — every few turns at most."
        )
    else:
        parts.append(
            "PERSONALISATION\n"
            "- No name is known. Speak directly; never invent a name; "
            "never say \"patient\" or \"user\"."
        )

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Layer 4 — Session-state (STEP 0)
# ---------------------------------------------------------------------------

def layer_session_state_instructions() -> str:
    return (
        "STEP 0 — CHECK PRIOR CONVERSATION FIRST\n"
        "- The new question is the priority — answer it; never redirect "
        "away from it.\n"
        "- If it links to earlier turns, briefly note the connection.\n"
        "- Mirror what they told you. Never restart the conversation."
    )


# ---------------------------------------------------------------------------
# Layer 5 — Retrieval grounding
# ---------------------------------------------------------------------------

def layer_retrieval_grounding() -> str:
    return (
        "CLINICAL KNOWLEDGE\n"
        "- A background snippets block may follow. Use it; don't quote "
        "chunks. If it doesn't match, use general medical knowledge.\n"
        "- Never mention retrieval, vectors, summaries, chunks, graph — "
        "speak as if you simply know."
    )


# ---------------------------------------------------------------------------
# Layer 6 — Tool instructions (reserved)
# ---------------------------------------------------------------------------

def layer_tool_instructions(tools: list | None = None) -> str:
    return ""


# ---------------------------------------------------------------------------
# Layer 7 — Formatting constraints (STEP 1/2/3 consultation flow)
# ---------------------------------------------------------------------------

def layer_formatting_constraints(*, query_type: str) -> str:
    classified = (query_type or "unknown").strip().lower()
    if classified not in _SUBSTANTIVE_QUERY_TYPES:
        return (
            f"OUTPUT SHAPE (query: {classified})\n"
            f"- This isn't a substantive clinical concern (greeting, "
            f"thanks, quick yes/no follow-up). Answer naturally in 1–2 "
            f"sentences. Do NOT force the interview flow onto small talk."
        )

    return (
        f"CONSULTATION FLOW (query: {classified})\n"
        f"STEP 1 — First mention of a symptom: don't diagnose. "
        f"Acknowledge, then ask ONE focused question. Every question MUST "
        f"state WHY in one clause (e.g. \"chest pain radiating to your "
        f"arm or with shortness of breath? — to tell cardiac from "
        f"muscular/respiratory\"). Never vague or random.\n"
        f"STEP 2 — Acknowledge each answer; ask ONE more if needed, with "
        f"its medical reasoning. Never multiple per turn. HARD CAP: at "
        f"most 5 follow-ups total — fewer if you have enough.\n"
        f"STEP 3 — Once enough (or at the 5-question cap): 2–3 likely "
        f"causes from the whole conversation, practical today-actions, "
        f"the doctor-confirmation line, a clear next step.\n"
        f"EMERGENCY EXCEPTION — life-threatening signs (see safety): "
        f"SKIP the interview, escalate immediately."
    )


# ---------------------------------------------------------------------------
# Composer
# ---------------------------------------------------------------------------

def compose_system_prompt(
    *,
    query_type: str,
    risk_level: str = "none",
    has_name: bool = False,
    tools: list | None = None,
) -> str:
    """
    Compose the layered system prompt for the clinical answer LLM.

    Args:
        query_type: Active task classification (e.g. ``symptom_query``,
            ``greeting``). Drives the formatting layer's branch.
        risk_level: One of ``none | low | medium | high | critical`` (from
            ``analysis.risk_level``). Surfaces a tone header at the top of
            the runtime layer when high or critical.
        has_name: ``True`` when the structured memory block contains a
            ``Patient name:`` line. Drives the personalisation layer's
            variant (greet-by-name vs no-name).
        tools: Reserved hook for future tool-calling. Currently unused.

    Returns:
        The fully assembled system prompt string with empty layers omitted.
    """
    layers = [
        layer_core_identity(),
        layer_safety_policy(),
        layer_runtime_modifiers(risk_level=risk_level, has_name=has_name),
        layer_session_state_instructions(),
        layer_retrieval_grounding(),
        layer_tool_instructions(tools),
        layer_formatting_constraints(query_type=query_type),
    ]
    return "\n\n".join(layer for layer in layers if layer)


__all__ = [
    "compose_system_prompt",
    "layer_core_identity",
    "layer_safety_policy",
    "layer_runtime_modifiers",
    "layer_session_state_instructions",
    "layer_retrieval_grounding",
    "layer_tool_instructions",
    "layer_formatting_constraints",
]
