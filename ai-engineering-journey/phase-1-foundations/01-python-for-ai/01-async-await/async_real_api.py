"""
async_real_api.py  (OPTIONAL)
=============================
Phase 1 > 01 Python for AI > 01 async/await

WHAT THIS FILE TEACHES
----------------------
The same patterns from async_llm_patterns.py, against a REAL model API:
  - an async SDK client (AsyncAnthropic) with its built-in retries + timeout
  - a Semaphore to stay under your account's rate limits
  - a TaskGroup to run all calls concurrently with fail-fast behaviour

REQUIRES (paid resource)
------------------------
An Anthropic API key with prepaid credits (console.anthropic.com).
This script makes 6 tiny calls -- well under one cent.
Every major provider (OpenAI, Gemini, ...) ships an equivalent async client,
so skipping this file loses nothing conceptually.

SETUP
-----
    uv add anthropic
    # macOS/Linux:
    export ANTHROPIC_API_KEY=sk-ant-...
    # Windows PowerShell:
    $env:ANTHROPIC_API_KEY="sk-ant-..."

    uv run async_real_api.py

Never hard-code or commit the key. (Config & secrets get their own lesson
later in this phase -- .env files, pydantic-settings, secret managers.)
"""

import asyncio
import time

# AsyncAnthropic is the async version of the client. Its methods are
# coroutines, so they must be awaited -- and they don't block the loop.
# (The sync `Anthropic` client would block the loop: see demo 6.)
from anthropic import AsyncAnthropic

# The client reads ANTHROPIC_API_KEY from the environment automatically.
# max_retries: the SDK itself retries 429 / 5xx / connection errors with
#   exponential backoff. Configure it here rather than writing your own retry
#   loop around it -- two retry layers multiply (3 x 3 = 9 attempts).
# timeout: seconds before a single request is abandoned.
client = AsyncAnthropic(max_retries=3, timeout=30.0)

# Module-level semaphore = a cap shared by every call in this process.
# Pick a value safely under your account tier's concurrency / RPM limits.
sem = asyncio.Semaphore(4)

# Sample input "documents" to summarise.
DOCS = [
    "Retrieval-augmented generation grounds model answers in fetched documents.",
    "Vector databases index embeddings for approximate nearest-neighbour search.",
    "Rerankers rescore retrieved candidates with a cross-encoder for precision.",
    "Evals measure whether a change to prompts or retrieval made things better.",
    "Agents loop: the model chooses a tool, observes the result, and decides again.",
    "Guardrails validate inputs and outputs against policy before they reach users.",
]


async def summarize(text: str) -> str:
    # Only 4 coroutines can be inside this block at once; the other 2 wait.
    async with sem:
        # messages.create is the Messages API call. Awaiting it suspends this
        # coroutine while the request is in flight, letting the others run.
        msg = await client.messages.create(
            # A small, fast, cheap model -- right for high-volume simple tasks.
            # Model names change over time; check docs.claude.com for current IDs.
            model="claude-haiku-5-5",
            # Hard cap on output tokens = a hard cap on cost and latency.
            max_tokens=60,
            # The conversation: a list of {"role", "content"} messages.
            messages=[{"role": "user", "content": f"Rewrite in 8 words or fewer:\n{text}"}],
        )
        # The response content is a list of blocks; the first is the text.
        return msg.content[0].text.strip()


async def main():
    t = time.perf_counter()
    # TaskGroup: if any call ultimately fails (after the SDK's retries),
    # the remaining calls are cancelled and the error is raised here.
    async with asyncio.TaskGroup() as tg:
        tasks = [tg.create_task(summarize(d)) for d in DOCS]
    # After the `async with` block exits, every task is guaranteed done,
    # so .result() returns immediately (no await needed).
    for d, task in zip(DOCS, tasks):
        print(f"- {task.result()}")
    print(f"\n{len(DOCS)} calls in {time.perf_counter() - t:.2f}s")


if __name__ == "__main__":
    asyncio.run(main())
