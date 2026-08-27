"""Lazy-loading FlagEmbedding reranker with idle-timeout GPU unload.

Ollama has no rerank API for a cross-encoder model like bge-reranker-v2-m3
(ARCHITECTURE.md section 1), so this loads FlagEmbedding directly as a
local process. The GPU is shared with other services on the prod box, so
the model is loaded lazily on first use and unloaded after
`idle_timeout_seconds` of inactivity rather than held in VRAM permanently -
confirmed end-to-end (cold load ~468s incl. one-time HF download, warm
load from cache ~1.6s, so the unload/reload cycle costs ~1.6s of latency,
not a user-visible stall).
"""

from __future__ import annotations

import asyncio
import logging
import time

logger = logging.getLogger(__name__)


class LazyReranker:
    def __init__(self, model_name: str, device: str, idle_timeout_seconds: int) -> None:
        self.model_name = model_name
        self.device = device
        self.idle_timeout_seconds = idle_timeout_seconds
        self._model = None
        self._lock = asyncio.Lock()
        self._last_used = 0.0
        self._idle_task: asyncio.Task | None = None

    async def rerank(self, query: str, passages: list[str]) -> list[float]:
        """Returns a relevance score per passage, same order as input.
        Higher is more relevant (bge-reranker-v2-m3's raw logit, not a
        probability - only meaningful for ranking, not thresholding)."""
        if not passages:
            return []

        await self._ensure_loaded()
        pairs = [(query, passage) for passage in passages]
        loop = asyncio.get_running_loop()
        scores = await loop.run_in_executor(None, self._model.compute_score, pairs)
        self._last_used = time.monotonic()

        if isinstance(scores, float):
            return [scores]
        return list(scores)

    async def _ensure_loaded(self) -> None:
        async with self._lock:
            if self._model is None:
                from FlagEmbedding import FlagReranker  # heavy import, deferred to first use

                start = time.monotonic()
                self._model = FlagReranker(self.model_name, use_fp16=True, devices=[self.device])
                logger.info("reranker_load_time_ms=%.1f", (time.monotonic() - start) * 1000)
            self._last_used = time.monotonic()
            if self._idle_task is None or self._idle_task.done():
                self._idle_task = asyncio.create_task(self._watch_idle())

    async def _watch_idle(self) -> None:
        while True:
            await asyncio.sleep(self.idle_timeout_seconds)
            async with self._lock:
                if self._model is None:
                    return
                if time.monotonic() - self._last_used < self.idle_timeout_seconds:
                    continue  # used again since we last checked - keep watching
                del self._model
                self._model = None
                import torch

                torch.cuda.empty_cache()
                logger.info("reranker_idle_unload=true")
                return
