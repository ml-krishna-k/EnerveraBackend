"""
Unit tests for the layered system-prompt composer.

These tests snapshot the *promises* of the original monolithic prompt so a
future edit to any single layer can't silently drop a rule. They also lock
in the runtime layer's branching behaviour (risk header on/off, with-name
vs no-name personalisation, substantive vs non-substantive formatting).

Pure-function, no network, no fixtures needed.
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


def test_core_identity_static():
    out = layer_core_identity()
    assert "warm, Indian conversational medical companion" in out
    assert "thoughtful friend who happens to be a clinician" in out


def test_safety_policy_probabilistic():
    out = layer_safety_policy()
    assert "probabilistic" in out
    assert "only a doctor can confirm" in out
    assert "Never make a definitive diagnosis" in out


def test_safety_policy_emergency_net():
    out = layer_safety_policy()
    assert "112/911" in out
    assert "A&E" in out


def test_runtime_personalisation_with_name():
    out = layer_runtime_modifiers(risk_level="none", has_name=True)
    assert "Hey Aarav" in out
    assert "sparingly" in out
    assert "PERSONALISATION" in out


def test_runtime_personalisation_no_name():
    out = layer_runtime_modifiers(risk_level="none", has_name=False)
    assert "Do NOT invent" in out
    assert '"patient"' in out and '"user"' in out
    assert "Hey Aarav" not in out


def test_runtime_risk_critical_surfaces_warning():
    out = layer_runtime_modifiers(risk_level="critical", has_name=False)
    assert "⚠️ CRITICAL" in out
    # The personalisation block still follows the risk header.
    assert "PERSONALISATION" in out


def test_runtime_risk_none_omits_header():
    out = layer_runtime_modifiers(risk_level="none", has_name=False)
    assert "⚠️" not in out
    assert "CRITICAL" not in out
    assert "Elevated risk" not in out


def test_retrieval_grounding_forbids_meta_leak():
    out = layer_retrieval_grounding()
    assert "Never mention retrieval" in out
    # All five meta-leak terms the upstream prompt forbids.
    for term in ("retrieval", "vectors", "summaries", "chunks", "graph"):
        assert term in out


def test_session_state_layer_mentions_memory_block():
    out = layer_session_state_instructions()
    assert "memory block" in out
    assert "Never restart the conversation" in out


def test_tool_instructions_empty_today():
    # The layer is a reserved hook for future tools — returns empty so the
    # composer skips it.
    assert layer_tool_instructions() == ""
    assert layer_tool_instructions(tools=[]) == ""


# ---------------------------------------------------------------------------
# Formatting layer — substantive vs non-substantive branches
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("query_type", [
    "symptom_query", "diagnosis_query", "diagnosis",
    "medication_query", "treatment_query", "drug_interaction",
    "guideline", "lab_interpretation", "prognosis", "unknown",
])
def test_formatting_substantive_demands_cause_action_why(query_type: str):
    out = layer_formatting_constraints(query_type=query_type)
    # The three substance pieces must all be named.
    assert "most likely cause" in out
    assert "concrete, specific" in out
    assert "reason WHY" in out


def test_formatting_substantive_forbids_labels():
    out = layer_formatting_constraints(query_type="symptom_query")
    # The forbidden words appear ONLY as forbidden examples — not as labels.
    assert "never as labeled sections" in out
    assert '"probable cause"' in out  # quoted as a forbidden example
    assert '"primary care"' in out
    assert '"justification"' in out
    # No real label headers in the prompt text itself.
    assert "\nPROBABLE CAUSE:" not in out
    assert "\nPRIMARY CARE:" not in out


def test_formatting_substantive_caps_followups():
    out = layer_formatting_constraints(query_type="symptom_query")
    assert "ONE clarifying question" in out
    assert "ask nothing" in out


def test_formatting_non_substantive_is_short_no_three_part():
    out = layer_formatting_constraints(query_type="greeting")
    # Non-substantive should NOT carry the cause/action/why scaffold.
    assert "most likely cause" not in out
    assert "(a)" not in out
    # Should explicitly note it's not a clinical concern.
    assert "1–2 sentences" in out
    assert "small talk" in out


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
    assert "warm, Indian conversational medical companion" in out  # L1
    assert "SAFETY" in out and "112/911" in out                    # L2
    assert "⚠️ CRITICAL" in out and "Hey Aarav" in out             # L3
    assert "SESSION STATE" in out                                  # L4
    assert "CLINICAL KNOWLEDGE" in out                             # L5
    # L6 is empty by design.
    assert "OUTPUT SHAPE" in out and "reason WHY" in out           # L7


def test_compose_skips_empty_layers_for_low_risk_no_name_greeting():
    out = compose_system_prompt(
        query_type="greeting",
        risk_level="none",
        has_name=False,
    )
    # Risk header is suppressed when risk_level is none.
    assert "⚠️ CRITICAL" not in out
    assert "Elevated risk" not in out
    # Personalisation still present, but in the no-name variant.
    assert "Do NOT invent" in out
    assert "Hey Aarav" not in out
    # Formatting is the short-form (non-substantive) branch.
    assert "1–2 sentences" in out
    assert "most likely cause" not in out


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
    # No kwargs other than query_type — should still produce a sensible prompt
    # (caller forgot to pass risk_level / has_name).
    out = compose_system_prompt(query_type="symptom_query")
    assert "warm, Indian conversational medical companion" in out
    assert "OUTPUT SHAPE" in out
    # Defaults: risk none, has_name False.
    assert "⚠️" not in out
    assert "Hey Aarav" not in out
    assert "Do NOT invent" in out
