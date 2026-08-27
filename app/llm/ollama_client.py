"""LLM generation client backed by Ollama's `/api/chat` endpoint. Model is
read from config (`OLLAMA_MODEL`), never hardcoded (PLAN.md section 4).
"""

from __future__ import annotations

import ollama


class OllamaGenerationClient:
    def __init__(self, base_url: str, model: str, num_ctx: int) -> None:
        self._client = ollama.AsyncClient(host=base_url)
        self.model = model
        self.num_ctx = num_ctx

    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = await self._client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            # Reasoning-capable local models (e.g. qwen3.5) otherwise spend
            # the entire output token budget on hidden thinking and return
            # empty content - confirmed on this project's own qwen3.5:9b
            # (done_reason="length", eval_count=4077, content=""). We want
            # the final structured answer, not exposed chain-of-thought.
            think=False,
            # Ollama defaults num_ctx far below what qwen3.5:9b actually
            # supports (262144) to save VRAM, regardless of the model's own
            # capability - confirmed responses truncating mid-answer once
            # project_chat.py's retrieval expansion (ARCHITECTURE.md
            # section 21) started packing more retrieved chunks into the
            # prompt than the Ollama default window could hold alongside a
            # full structured answer. Explicit, not implicit.
            options={"num_ctx": self.num_ctx},
        )
        return response.message.content
