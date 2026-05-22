"""
Unit tests for the layered system-prompt composer.

These tests lock in the *promises* of the gastroenterology-consultation
prompt style — one-question-at-a-time interview flow, emergency-number
restraint, doctor-confirmation disclaimer, prior-context handling — plus
the composer's branching behaviour (risk header on/off, with-name vs
no-name personalisation, substantive vs non-substantive formatting).

Assertions use `.lower()` substring matching where exact casing is not the
contract (case-sensitive only when capitalisation carries weight, like
"SAFETY" headers and ALL-CAPS rule emphasis). The compressed layer text
fits a 400-token budget; tests target rule presence, not exact wording.
"""

from __future__ import annotations

import pytest

from app.services.orchestration.prompt_layers import (
    compose_system_prompt,
    layer_core_identity,
    layer_formatting_constraints,
    layer_retrieval_grounding,
    layer_runtime_modifiers,
    layer_safety_policy,
    layer_session_state_instructions,
    layer_tool_instructions,
)


# ---------------------------------------------------------------------------
# Per-layer assertions
# ---------------------------------------------------------------------------


def test_core_identity_gastroenterology_persona():
    out = layer_core_identity()
    assert "warm, empathetic gastroenterology assistant" in out
    assert "knowledgeable friend" in out
    assert "real doctor" in out
    assert "feel heard first, then helped" in out


def test_safety_policy_probabilistic_and_doctor_disclaimer():
    out = layer_safety_policy().lower()
    assert "probabilistic language" in out
    assert "only a doctor can properly examine and confirm this" in out
    assert "definitive diagnosis" in out


def test_safety_policy_emergency_number_restraint():
    out = layer_safety_policy()
    # Emergency numbers are gated to life-threatening only.
    assert "112" in out and "102" in out and "108" in out
    # The hard rule — never for routine complaints — must be in caps.
    assert "NEVER show emergency numbers for routine" in out
    # Named routine GI complaints that must NOT trigger emergency numbers.
    for routine in ("bloating", "acidity", "burping", "constipation"):
        assert routine in out


def test_safety_policy_lists_life_threatening_red_flags():
    out = layer_safety_policy()
    # Specific red-flag examples that DO warrant emergency numbers.
    for red_flag in (
        "vomiting blood",
        "severe chest pain",
        "can't breathe",
        "collapse",
        "black tarry stool",
    ):
        assert red_flag in out


def test_runtime_personalisation_with_name():
    out = layer_runtime_modifiers(risk_level="none", has_name=True)
    assert "Hey Aarav" in out
    assert "sparingly" in out
    assert "PERSONALISATION" in out


def test_runtime_personalisation_no_name():
    out = layer_runtime_modifiers(risk_level="none", has_name=False).lower()
    assert "no name is known" in out
    assert "never invent" in out
    assert '"patient"' in out and '"user"' in out
    assert "hey aarav" not in out


def test_runtime_risk_critical_surfaces_warning():
    out = layer_runtime_modifiers(risk_level="critical", has_name=False)
    assert "⚠️ CRITICAL" in out
    # Critical block tells the LLM to skip the interview flow.
    assert "skip the interview flow" in out.lower() or \
           "SKIP the interview" in out
    # Personalisation block still follows the risk header.
    assert "PERSONALISATION" in out


def test_runtime_risk_none_omits_header():
    out = layer_runtime_modifiers(risk_level="none", has_name=False)
    assert "⚠️" not in out
    assert "CRITICAL" not in out
    assert "Elevated risk" not in out


def test_session_state_step_zero_prior_context():
    out = layer_session_state_instructions()
    # STEP 0 framing — the LLM must check prior turns first.
    assert "STEP 0" in out
    assert "prior conversation" in out.lower()
    # New question takes priority, no redirecting away from it.
    assert "new question is the priority" in out.lower()
    assert "redirect" in out.lower()
    # Conversation is never restarted.
    assert "never restart" in out.lower()


def test_retrieval_grounding_forbids_meta_leak():
    out = layer_retrieval_grounding()
    assert "Never mention retrieval" in out
    # Each meta-leak term forbidden by name.
    for term in ("retrieval", "vectors", "summaries", "chunks", "graph"):
        assert term in out


def test_tool_instructions_empty_today():
    assert layer_tool_instructions() == ""
    assert layer_tool_instructions(tools=[]) == ""


# ---------------------------------------------------------------------------
# Formatting layer — substantive (consultation flow) vs non-substantive
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("query_type", [
    "symptom_query", "diagnosis_query", "diagnosis",
    "medication_query", "treatment_query", "drug_interaction",
    "guideline", "lab_interpretation", "prognosis", "unknown",
])
def test_formatting_substantive_uses_consultation_flow(query_type: str):
    out = layer_formatting_constraints(query_type=query_type)
    assert "CONSULTATION FLOW" in out
    assert "STEP 1" in out
    assert "STEP 2" in out
    assert "STEP 3" in out


def test_formatting_substantive_one_question_at_a_time():
    out = layer_formatting_constraints(query_type="symptom_query")
    lo = out.lower()
    # The central rule of the new style.
    assert "one focused question" in lo
    # Never multiple questions in a single turn.
    assert "never multiple per turn" in lo or "never multiple" in lo
    # Step 1 explicitly forbids two/three questions in one turn.
    assert "one only" in lo or "one question only" in lo


def test_formatting_substantive_step1_examples_present():
    out = layer_formatting_constraints(query_type="symptom_query")
    # Example dimensions a real GI doctor probes first.
    assert "duration" in out
    assert "triggers" in out
    assert "severity" in out
    # The 1-10 severity scale is referenced.
    assert "1–10" in out or "1-10" in out


def test_formatting_substantive_step3_synthesis_rules():
    out = layer_formatting_constraints(query_type="symptom_query")
    lo = out.lower()
    # Step 3 synthesis pieces.
    assert "2–3 likely causes" in out
    assert "today" in lo  # "practical today-actions" or similar
    # Doctor-confirmation line is referenced (the actual disclaimer string
    # itself lives in the safety layer).
    assert "doctor-confirmation" in lo or "doctor confirmation" in lo


def test_formatting_substantive_carries_emergency_exception():
    out = layer_formatting_constraints(query_type="symptom_query")
    assert "EMERGENCY EXCEPTION" in out
    assert "skip" in out.lower()
    assert "escalate" in out.lower() or "emergency services" in out.lower()


def test_formatting_non_substantive_is_short():
    out = layer_formatting_constraints(query_type="greeting")
    # Non-substantive — no consultation flow, just 1-2 sentences.
    assert "CONSULTATION FLOW" not in out
    assert "1–2 sentences" in out
    # No STEP scaffolding for greetings.
    assert "STEP 1" not in out
    assert "STEP 2" not in out


# ---------------------------------------------------------------------------
# Composer — joins, skips, idempotent
# ---------------------------------------------------------------------------


def test_compose_joins_all_layers_for_substantive_with_name_critical():
    out = compose_system_prompt(
        query_type="symptom_query",
        risk_level="critical",
        has_name=True,
    )
    # Markers from every non-empty layer must appear in the composed prompt.
    assert "gastroenterology assistant" in out                     # L1
    assert "SAFETY" in out and "112" in out                        # L2
    assert "⚠️ CRITICAL" in out and "Hey Aarav" in out             # L3
    assert "STEP 0" in out                                         # L4
    assert "CLINICAL KNOWLEDGE" in out                             # L5
    # L6 is empty by design.
    assert "CONSULTATION FLOW" in out and "STEP 1" in out          # L7


def test_compose_skips_empty_layers_for_low_risk_no_name_greeting():
    out = compose_system_prompt(
        query_type="greeting",
        risk_level="none",
        has_name=False,
    )
    # Risk header is suppressed when risk_level is none.
    assert "⚠️ CRITICAL" not in out
    assert "Elevated risk" not in out
    # Personalisation still present but in the no-name variant.
    assert "no name is known" in out.lower()
    assert "Hey Aarav" not in out
    # Greeting → short non-substantive formatting branch.
    assert "1–2 sentences" in out
    assert "CONSULTATION FLOW" not in out


def test_compose_idempotent_pure_function():
    args = dict(query_type="symptom_query", risk_level="low", has_name=True)
    a = compose_system_prompt(**args)
    b = compose_system_prompt(**args)
    assert a == b


def test_compose_no_blank_line_runs():
    out = compose_system_prompt(
        query_type="symptom_query",
        risk_level="none",
        has_name=False,
    )
    # Layers are joined with "\n\n"; no triple-newline runs should appear.
    assert "\n\n\n" not in out


def test_compose_defaults_safe():
    # No kwargs other than query_type — defaults risk=none, has_name=False.
    out = compose_system_prompt(query_type="symptom_query")
    assert "gastroenterology assistant" in out
    assert "CONSULTATION FLOW" in out
    assert "⚠️" not in out
    assert "Hey Aarav" not in out
    assert "no name is known" in out.lower()


# ---------------------------------------------------------------------------
# Budget check — informational, not a hard fail
# ---------------------------------------------------------------------------


def test_compose_typical_path_fits_token_budget():
    """
    The composed prompt for the substantive-no-name-no-risk path should fit
    within roughly 400 tokens (~1600 chars, conservative 4-chars/token).
    Allow a small slack — the budget is a soft target, not a hard cap (no
    automatic truncation exists in the active code path).
    """
    out = compose_system_prompt(
        query_type="symptom_query",
        risk_level="none",
        has_name=False,
    )
    chars = len(out)
    # Soft cap: 1800 chars (~450 tokens). If we drift over this, compression
    # has eroded — re-tighten the layer text.
    assert chars <= 1800, (
        f"Composed prompt is {chars} chars (~{chars // 4} tokens); "
        f"compression has drifted, retighten layer text."
    )
