"""
AsyncOrchestrator — async-native port of GraphRAGPipeline.run().

Stage-by-stage parity with the existing sync pipeline:

    -2  Session memory load (async — SessionManager directly)
    -1  Medical query analyzer (async — Gemini JSON mode)
     0  Routing decision (pure)
     1  Pinecone retrieval (sync client → asyncio.to_thread)
     2  Entity extraction (pure / CPU)
     3  Neo4j traversal (sync driver → asyncio.to_thread)
   3.5  Episodic memory context (async-native)
     4  Gemini answer (async non-streaming; streaming path in stream())
     5  Episodic ingest (async-native, fire-and-forget)
    5b  Session save (async)

The orchestrator never holds Pinecone/Neo4j connections itself — it borrows
them from AppContainer. State is request-scoped (request_id, session, ...).
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, AsyncIterator

from app.services.memory.session import (
    assemble_memory_payload,
    build_retrieval_query,
    load_session,
    save_after_turn,
)
from graphrag.query_understanding import (
    QueryType,
    RoutingMode,
    decide_routing,
    get_config,
    is_trivial_input,
)

if TYPE_CHECKING:
    from app.container import AppContainer

logger = logging.getLogger(__name__)


@dataclass
class ChatResult:
    answer: str
    session_id: str
    request_id: str
    analysis: dict[str, Any] | None = None
    timing_ms: dict[str, int] = field(default_factory=dict)
    routing: dict[str, Any] = field(default_factory=dict)
    followup_questions: list[str] = field(default_factory=list)


class AsyncOrchestrator:
    def __init__(self, container: "AppContainer") -> None:
        self._c = container

    # ------------------------------------------------------------------
    # Public — non-streaming
    # ------------------------------------------------------------------

    async def run(
        self,
        *,
        query: str,
        session_id: str,
        user_id: str | None,
        request_id: str,
    ) -> ChatResult:
        timing: dict[str, int] = {}
        t0 = time.monotonic()

        # Stage -2: Session memory
        with _Stage("session_load", timing):
            bundle = await load_session(self._c.session_manager, session_id)

        session = bundle.session
        wm = bundle.working_memory
        memory_query_text = build_retrieval_query(query, wm)
        analyzer_input = memory_query_text if (wm.turn_count or wm.has_summary) else query

        # Stage -1: Gatekeeper analyzer
        trivial_skip = is_trivial_input(query) and wm.turn_count > 0
        if trivial_skip:
            analysis: dict[str, Any] = {}
        else:
            with _Stage("analyze", timing):
                analysis = await self._c.analyzer.aanalyze(analyzer_input)

        # Short-circuit: refuse / emergency_redirect
        final_action = (analysis or {}).get("final_action")
        if analysis and "error" not in analysis and final_action in {"refuse", "emergency_redirect"}:
            msg = _canned_message(final_action)
            await save_after_turn(
                self._c.session_manager,
                session=session,
                user_query=query,
                assistant_answer=msg,
                analysis=analysis,
                query_type="emergency" if final_action == "emergency_redirect" else "unknown",
            )
            timing["total"] = int((time.monotonic() - t0) * 1000)
            return ChatResult(
                answer=msg,
                session_id=session_id,
                request_id=request_id,
                analysis=analysis,
                timing_ms=timing,
                routing={"mode": "short_circuit", "intent": final_action},
            )

        followup_questions = self._extract_followups(analysis)

        # Rewritten query (if analyzer suggested one)
        rewritten = (analysis or {}).get("rewritten_query")
        active_query = (
            rewritten.strip() if rewritten and rewritten.strip() and rewritten != query else query
        )

        # Stage 0: Routing
        routing_mode, query_type = decide_routing(
            analysis=analysis, wm=wm, raw_query=query
        )
        cfg = get_config(query_type)
        intent_str = (analysis or {}).get("intent") or "unknown"
        vector_top_k, reranker_top_k, graph_hops = _route_budget(routing_mode, cfg)

        # Stage 1: Pinecone (sync client → thread)
        retrieval_query_text = build_retrieval_query(active_query, wm)
        if vector_top_k > 0:
            with _Stage("vector_retrieve", timing):
                matches = await asyncio.to_thread(
                    self._c.vector_retriever.retrieve,
                    retrieval_query_text,
                    vector_top_k,
                    reranker_top_k,
                )
        else:
            matches = []

        # Stage 2: Entity extraction (pure)
        from graphrag.processors.entity_processor import EntityProcessor  # local: keep import light
        processor = EntityProcessor()
        vector_context_str, extracted_entities, _ = processor.process_matches(
            matches,
            priority_entity_types=cfg.priority_entity_types,
            boost_drug_pairs=cfg.boost_drug_pairs,
            query=retrieval_query_text,
        )

        # Stage 3: Neo4j (sync driver → thread)
        if graph_hops > 0 and extracted_entities:
            with _Stage("graph_retrieve", timing):
                graph_lines = await asyncio.to_thread(
                    self._c.kg_retriever.retrieve_relations,
                    extracted_entities,
                    graph_hops,
                    20,
                )
            graph_context_str = (
                "\n".join(f"- {g}" for g in graph_lines) if graph_lines else "No relevant relations found."
            )
        else:
            graph_context_str = ""

        # Stage 3.5: Episodic memory context (async-native)
        episodic_context_str = ""
        if user_id and self._c.episodic is not None:
            with _Stage("episodic_context", timing):
                episodic_context_str = await self._load_episodic_context(
                    user_id=user_id, query_text=retrieval_query_text
                )

        # Stage 4: LLM answer
        memory_payload = assemble_memory_payload(
            wm=wm,
            user_query=query,
            query_type=intent_str,
            goal=cfg.goal,
            vector_context=vector_context_str,
            graph_context=graph_context_str,
        )
        combined_memory = memory_payload.memory_context
        if episodic_context_str:
            combined_memory = episodic_context_str.strip() + "\n\n" + combined_memory

        with _Stage("llm", timing):
            answer = await self._answer_async(
                query=query,
                memory_context=combined_memory,
                conversation_history=memory_payload.conversation_context,
                vector_context=vector_context_str,
                graph_context=graph_context_str,
                query_type=intent_str,
                goal=cfg.goal,
            )

        if followup_questions and answer:
            followup_block = (
                "\n\n---\n💬 **To help me give you a more precise answer next time, "
                "could you also share:**\n"
                + "\n".join(f"- {q}" for q in followup_questions)
            )
            answer = answer + followup_block

        # Stage 5: Episodic ingest (fire-and-forget; never blocks response)
        if user_id and self._c.episodic is not None:
            asyncio.create_task(self._ingest_episodic_safe(user_id=user_id, utterance=query))

        # Stage 5b: Session save
        with _Stage("session_save", timing):
            await save_after_turn(
                self._c.session_manager,
                session=session,
                user_query=query,
                assistant_answer=answer or "",
                analysis=analysis or {},
                query_type=query_type.value,
            )

        timing["total"] = int((time.monotonic() - t0) * 1000)

        return ChatResult(
            answer=answer or "",
            session_id=session_id,
            request_id=request_id,
            analysis=analysis or None,
            timing_ms=timing,
            routing={
                "mode": routing_mode.value,
                "intent": intent_str,
                "query_type": query_type.value,
                "vector_top_k": vector_top_k,
                "graph_hops": graph_hops,
            },
            followup_questions=followup_questions,
        )

    # ------------------------------------------------------------------
    # Public — streaming (filled in by phase 3)
    # ------------------------------------------------------------------

    async def stream(
        self,
        *,
        query: str,
        session_id: str,
        user_id: str | None,
        request_id: str,
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Yield SSE-shaped events as the pipeline progresses.

        Event types:
            meta   — pipeline metadata (routing, timing of pre-LLM stages)
            chunk  — one piece of streamed model output
            done   — final event with total timing + assistant_answer
            error  — terminal error event (also ends the stream)

        The pre-LLM stages run exactly as in run(); only Stage 4 changes
        from a single await to an async iterator. After the stream ends we
        run session_save + fire-and-forget episodic ingest just like run().
        """
        from app.services.llm.streaming import stream_gemini_tokens
        from graphrag.config.settings import settings as cfg
        from graphrag.llm.gemini_client import DEFAULT_MODEL

        timing: dict[str, int] = {}
        t0 = time.monotonic()

        # ------------------------------------------------------------------
        # Pre-LLM stages (identical to run())
        # ------------------------------------------------------------------
        try:
            with _Stage("session_load", timing):
                bundle = await load_session(self._c.session_manager, session_id)
            session = bundle.session
            wm = bundle.working_memory
            memory_query_text = build_retrieval_query(query, wm)
            analyzer_input = memory_query_text if (wm.turn_count or wm.has_summary) else query

            trivial_skip = is_trivial_input(query) and wm.turn_count > 0
            if trivial_skip:
                analysis: dict[str, Any] = {}
            else:
                with _Stage("analyze", timing):
                    analysis = await self._c.analyzer.aanalyze(analyzer_input)

            final_action = (analysis or {}).get("final_action")
            if analysis and "error" not in analysis and final_action in {"refuse", "emergency_redirect"}:
                msg = _canned_message(final_action)
                await save_after_turn(
                    self._c.session_manager,
                    session=session,
                    user_query=query,
                    assistant_answer=msg,
                    analysis=analysis,
                    query_type="emergency" if final_action == "emergency_redirect" else "unknown",
                )
                yield {"type": "chunk", "data": msg}
                timing["total"] = int((time.monotonic() - t0) * 1000)
                yield {"type": "done", "timing_ms": timing}
                return

            followup_questions = self._extract_followups(analysis)
            rewritten = (analysis or {}).get("rewritten_query")
            active_query = (
                rewritten.strip() if rewritten and rewritten.strip() and rewritten != query else query
            )

            routing_mode, query_type = decide_routing(
                analysis=analysis, wm=wm, raw_query=query
            )
            route_cfg = get_config(query_type)
            intent_str = (analysis or {}).get("intent") or "unknown"
            vector_top_k, reranker_top_k, graph_hops = _route_budget(routing_mode, route_cfg)

            retrieval_query_text = build_retrieval_query(active_query, wm)
            if vector_top_k > 0:
                with _Stage("vector_retrieve", timing):
                    matches = await asyncio.to_thread(
                        self._c.vector_retriever.retrieve,
                        retrieval_query_text,
                        vector_top_k,
                        reranker_top_k,
                    )
            else:
                matches = []

            from graphrag.processors.entity_processor import EntityProcessor
            processor = EntityProcessor()
            vector_context_str, extracted_entities, _ = processor.process_matches(
                matches,
                priority_entity_types=route_cfg.priority_entity_types,
                boost_drug_pairs=route_cfg.boost_drug_pairs,
                query=retrieval_query_text,
            )

            if graph_hops > 0 and extracted_entities:
                with _Stage("graph_retrieve", timing):
                    graph_lines = await asyncio.to_thread(
                        self._c.kg_retriever.retrieve_relations,
                        extracted_entities, graph_hops, 20,
                    )
                graph_context_str = (
                    "\n".join(f"- {g}" for g in graph_lines)
                    if graph_lines else "No relevant relations found."
                )
            else:
                graph_context_str = ""

            episodic_context_str = ""
            if user_id and self._c.episodic is not None:
                with _Stage("episodic_context", timing):
                    episodic_context_str = await self._load_episodic_context(
                        user_id=user_id, query_text=retrieval_query_text
                    )

            memory_payload = assemble_memory_payload(
                wm=wm,
                user_query=query,
                query_type=intent_str,
                goal=route_cfg.goal,
                vector_context=vector_context_str,
                graph_context=graph_context_str,
            )
            combined_memory = memory_payload.memory_context
            if episodic_context_str:
                combined_memory = episodic_context_str.strip() + "\n\n" + combined_memory

            # Tell the client what's about to happen so a UI can show status.
            yield {
                "type": "meta",
                "data": {
                    "routing": {
                        "mode": routing_mode.value,
                        "intent": intent_str,
                        "query_type": query_type.value,
                    },
                    "timing_ms": dict(timing),
                },
            }

            # ------------------------------------------------------------------
            # Stage 4: streaming LLM answer
            # ------------------------------------------------------------------
            system_prompt, user_prompt = _compose_answer_prompts(
                query=query,
                memory_context=combined_memory,
                conversation_history=memory_payload.conversation_context,
                vector_context=vector_context_str,
                graph_context=graph_context_str,
                query_type=intent_str,
            )

            llm_t0 = time.monotonic()
            answer_chunks: list[str] = []
            async for piece in stream_gemini_tokens(
                model=cfg.ANSWER_MODEL or DEFAULT_MODEL,
                system_instruction=system_prompt,
                user_prompt=user_prompt,
                temperature=0.2,
            ):
                answer_chunks.append(piece)
                yield {"type": "chunk", "data": piece}
            timing["llm"] = int((time.monotonic() - llm_t0) * 1000)

            answer = "".join(answer_chunks)
            if followup_questions and answer:
                followup_block = (
                    "\n\n---\n💬 **To help me give you a more precise answer next time, "
                    "could you also share:**\n"
                    + "\n".join(f"- {q}" for q in followup_questions)
                )
                yield {"type": "chunk", "data": followup_block}
                answer = answer + followup_block

            # ------------------------------------------------------------------
            # Post-stream: ingest + session save (don't block the done event)
            # ------------------------------------------------------------------
            if user_id and self._c.episodic is not None:
                asyncio.create_task(
                    self._ingest_episodic_safe(user_id=user_id, utterance=query)
                )

            with _Stage("session_save", timing):
                await save_after_turn(
                    self._c.session_manager,
                    session=session,
                    user_query=query,
                    assistant_answer=answer,
                    analysis=analysis or {},
                    query_type=query_type.value,
                )

            timing["total"] = int((time.monotonic() - t0) * 1000)
            yield {"type": "done", "timing_ms": timing}

        except Exception as exc:
            logger.exception("Streaming pipeline failed: %s", exc)
            yield {"type": "error", "error": {"code": "PIPELINE_ERROR", "message": str(exc)}}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _answer_async(
        self,
        *,
        query: str,
        memory_context: str,
        conversation_history: str,
        vector_context: str,
        graph_context: str,
        query_type: str,
        goal: str,
    ) -> str:
        """
        Non-streaming Gemini answer. Reuses GeminiLLM's prompt assembly but
        bypasses the sync stdout-printing path.
        """
        from graphrag.llm.gemini_client import DEFAULT_MODEL, generate_text_async
        from graphrag.config.settings import settings as cfg

        system_prompt, user_prompt = _compose_answer_prompts(
            query=query,
            memory_context=memory_context,
            conversation_history=conversation_history,
            vector_context=vector_context,
            graph_context=graph_context,
            query_type=query_type,
        )
        model = cfg.ANSWER_MODEL or DEFAULT_MODEL
        try:
            return await generate_text_async(
                user_prompt,
                model=model,
                system_instruction=system_prompt,
                temperature=0.2,
            )
        except Exception as exc:
            logger.exception("LLM answer failed: %s", exc)
            return ""

    async def _load_episodic_context(self, *, user_id: str, query_text: str) -> str:
        """Best-effort episodic context block; empty string on any failure."""
        try:
            from episodic.schemas.retrieval import RetrievalRequest
            block = await self._c.episodic.context_pipeline.build(
                RetrievalRequest(user_id=user_id, query_text=query_text)
            )
            return block.rendered_prompt or ""
        except Exception as exc:
            logger.warning("Episodic context load failed: %s", exc)
            return ""

    async def _ingest_episodic_safe(self, *, user_id: str, utterance: str) -> None:
        try:
            await self._c.episodic.ingest_pipeline.run(user_id=user_id, utterance=utterance)
        except Exception as exc:
            logger.warning("Episodic ingest failed: %s", exc)

    @staticmethod
    def _extract_followups(analysis: dict[str, Any] | None) -> list[str]:
        if not analysis or not analysis.get("needs_followup"):
            return []
        raw = analysis.get("followup_questions") or []
        # Hard cap: ≤1 question per turn (project contract).
        return [q for q in raw[:1] if q]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _Stage:
    """Record `stage`: duration_ms in the timing dict, exception-safe."""

    def __init__(self, name: str, sink: dict[str, int]) -> None:
        self._name = name
        self._sink = sink
        self._t0 = 0.0

    def __enter__(self) -> "_Stage":
        self._t0 = time.monotonic()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._sink[self._name] = int((time.monotonic() - self._t0) * 1000)


def _route_budget(mode: RoutingMode, cfg) -> tuple[int, int, int]:
    if mode == RoutingMode.NO_RETRIEVAL:
        return (0, 0, 0)
    if mode == RoutingMode.MEMORY_FIRST:
        return (3, 3, 0)
    return (cfg.vector_top_k, cfg.reranker_top_k, cfg.graph_hops)


def _canned_message(final_action: str) -> str:
    if final_action == "refuse":
        return (
            "I can only answer healthcare-related questions. Please ask a "
            "medical question."
        )
    return (
        "🚨 EMERGENCY: Your symptoms sound like a serious emergency. Please "
        "call emergency services (112 / 911) immediately or go to the nearest "
        "hospital."
    )


def _compose_answer_prompts(
    *,
    query: str,
    memory_context: str,
    conversation_history: str,
    vector_context: str,
    graph_context: str,
    query_type: str,
) -> tuple[str, str]:
    """
    Mirrors the prompts in graphrag.llm.gemini_llm.GeminiLLM.generate_response.
    Lifted here so the async path doesn't need GeminiLLM's streaming-only
    machinery.
    """
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
USER QUESTION: {query}

=== STRUCTURED CLINICAL MEMORY ===
{memory_context}

=== RECENT CONVERSATION ===
{conversation_history}

=== RETRIEVED MEDICAL CONTEXT ===
{vector_context}

=== GRAPH RELATIONS ===
{graph_context}
"""
    return system_prompt, user_prompt
