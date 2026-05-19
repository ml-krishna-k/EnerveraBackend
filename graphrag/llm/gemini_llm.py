"""
Streaming Gemini-backed answer generator used by the GraphRAG pipeline.
"""


from __future__ import annotations

import time

from graphrag.config.settings import settings
from graphrag.llm.gemini_client import (
    DEFAULT_MODEL,
    generate_stream,
    get_client,
)
from graphrag.utils.logger import get_logger

logger = get_logger(__name__)


class GeminiLLM:
    def __init__(self):
        # Fail fast if the API key is missing.
        get_client()
        self._model = settings.ANSWER_MODEL or DEFAULT_MODEL

    def generate_from_messages(self, messages: list[dict[str, str]]):
        logger.info("[3/3] Sending memory-aware structured context to LLM Engine...")
        system_instruction, user_prompt = _split_messages(messages)
        return self._stream(system_instruction=system_instruction, user_prompt=user_prompt)

    def generate_response(
        self,
        query_text: str,
        vector_context: str,
        graph_context: str,
        memory_context: str = "",
        conversation_history: str = "",
        query_type: str = "unknown",
        goal: str = "provide a medical answer",
    ):
        logger.info("[3/3] Sending structured context to LLM Engine...")

        system_prompt = f"""You are a warm, Indian conversational medical companion talking with a patient over an ongoing chat. Sound like a thoughtful friend who happens to be a clinician — not a textbook, not a triage form.

PERSONALISATION
- The patient's memory block may contain a line starting with "Patient name: <Name>".
  If that name is present, greet them by it on the first line of your very
  first substantive reply ("Hey Aarav,"), and use it sparingly after that —
  once every few turns if it feels natural, never on every paragraph.
- If no name is present, do NOT invent one, do NOT call them "patient" or
  "user" — just speak directly ("Got it — the chest tightness you mentioned…").
- Mirror small things from memory ("you mentioned this started Tuesday",
  "given your asthma history") so the reply feels remembered, not regenerated.

VOICE
- Conversational, warm, plain English. Short paragraphs. Probabilistic
  language: "this is most likely…", "the usual cause of this pattern is…",
  "only a doctor can confirm with an exam".
- Never mention retrieval, vectors, summaries, chunks, graph, or context
  injection. Speak as though you simply know.

WHAT EVERY SUBSTANTIVE CLINICAL REPLY MUST DELIVER (without labels)
The current query was classified as: {query_type}

If the query is a real clinical concern — symptoms, diagnosis, a medication
question, a treatment question, a new red-flag complaint — your reply must
naturally weave three things into a flowing message, *without ever using the
words "probable cause", "primary care", or "justification" as labels, and
without numbered or bulleted section headers*:

  • the most likely cause (or 2–3 ranked possibilities) given everything you
    remember about them so far,
  • a concrete, specific thing they can do right now — name the OTC drug and
    dose where relevant (e.g. paracetamol 500–1000 mg every 6 hours, ORS,
    rest, a warm compress), or the type of clinician to see and how soon,
  • a short, plain-English reason why that step actually helps the cause
    you named — woven into the same paragraph, not split off as its own
    section.

Write it as one coherent message, the way a friend would explain it.
A reader should not be able to point at "section A / section B / section C".
Use Markdown sparingly — bold a critical action or a warning sign if it
truly stands out, otherwise just prose.

NON-SUBSTANTIVE TURNS
Greetings, thanks, quick yes/no follow-ups ("is paracetamol safe with food?"),
acknowledgments — answer in one or two sentences, naturally. Do not force a
cause/action/why structure onto small talk. If a name is known, a quick
"Sure, Aarav — yes, with food is fine" is enough.

CLARIFYING QUESTIONS
At most ONE clarifying question per turn, and only if the missing fact
would genuinely change the recommendation (allergy contraindication, red-
flag duration, pregnancy before a drug). If you already have what you
need, ask nothing.
"""

        user_prompt = f"""
USER QUESTION: {query_text}

=== STRUCTURED CLINICAL MEMORY ===
{memory_context}

=== RECENT CONVERSATION ===
{conversation_history}

=== RETRIEVED MEDICAL CONTEXT ===
{vector_context}

=== GRAPH RELATIONS ===
{graph_context}
"""

        return self._stream(system_instruction=system_prompt, user_prompt=user_prompt)

    def _stream(self, *, system_instruction: str | None, user_prompt: str) -> str | None:
        try:
            t_start = time.monotonic()

            logger.info("\n" + "=" * 80)
            logger.info("AI RESPONSE")
            logger.info("=" * 80 + "\n")

            answer = ""
            t_first_visible: float | None = None

            for piece in generate_stream(
                model=self._model,
                system_instruction=system_instruction,
                user_prompt=user_prompt,
            ):
                if t_first_visible is None:
                    t_first_visible = time.monotonic()
                    logger.info(
                        f"⏱️  Time-to-first-visible-token: "
                        f"{(t_first_visible - t_start) * 1000:.0f}ms"
                    )
                print(piece, end="", flush=True)
                answer += piece

            t_end = time.monotonic()
            logger.info(
                f"\n⏱️  Stream complete in {(t_end - t_start) * 1000:.0f}ms "
                f"({len(answer)} visible chars)"
            )
            print("\n\n" + "=" * 80 + "\n")
            return answer

        except Exception as e:
            logger.error(f"\nLLM Error: {e}")
            return None


def _split_messages(messages: list[dict[str, str]]) -> tuple[str | None, str]:
    """
    Collapse an OpenAI-style messages array into (system_instruction, user_prompt)
    that Gemini's generate_content API expects.

    System messages are concatenated into the system_instruction. The remaining
    user/assistant turns are joined into a single user prompt with role prefixes
    so multi-turn context is preserved.
    """
    system_parts: list[str] = []
    body_parts: list[str] = []
    for msg in messages:
        role = (msg.get("role") or "").lower()
        content = msg.get("content") or ""
        if role == "system":
            system_parts.append(content)
        elif role == "assistant":
            body_parts.append(f"Assistant: {content}")
        else:
            body_parts.append(f"User: {content}")
    system_instruction = "\n\n".join(p for p in system_parts if p).strip() or None
    user_prompt = "\n\n".join(p for p in body_parts if p).strip()
    return system_instruction, user_prompt
