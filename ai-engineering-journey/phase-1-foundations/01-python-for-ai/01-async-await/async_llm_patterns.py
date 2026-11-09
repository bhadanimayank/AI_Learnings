"""
async_llm_patterns.py
=====================
Phase 1 > 01 Python for AI > 01 async/await

WHAT THIS FILE TEACHES
----------------------
Seven runnable demos of how async/await is used in real AI/LLM systems:

  1. Why async at all       -> I/O-bound fan-out (sequential vs concurrent)
  2. Coroutines are lazy    -> a coroutine does nothing until awaited / scheduled
  3. Unbounded fan-out      -> firing everything at once triggers 429 rate limits
  4. Production pattern     -> Semaphore + timeout + exponential backoff with jitter
  5. Structured concurrency -> TaskGroup cancels sibling tasks when one fails
  6. Blocking the loop      -> the #1 async bug in AI services, and the fix
  7. as_completed           -> process results in the order they finish

HOW TO RUN
----------
    uv run async_llm_patterns.py

Requirements: Python 3.11+ (TaskGroup, asyncio.timeout and except* need 3.11).
No third-party packages. No API key. Costs nothing.

WHY A FAKE CLIENT?
------------------
Real model APIs (OpenAI, Anthropic, Gemini, a self-hosted vLLM server...) all
behave the same way from your code's point of view:
  - each call takes a variable, fairly long time (network + inference)
  - the provider caps how many requests you may have in flight (-> HTTP 429)
  - some calls fail transiently (timeouts, 5xx, dropped connections)
FakeLLM simulates all three so you can trigger each failure mode on demand.
"""

# --- Imports ---------------------------------------------------------------
# asyncio : Python's built-in event loop + coroutine/task machinery.
# random  : simulates variable latency and random failures.
# time    : perf_counter() is a high-resolution clock for measuring durations.
import asyncio
import random
import time

# Fixing the seed makes the "random" latencies/failures repeatable between
# runs, so your output stays comparable while you experiment.
random.seed(7)


# ===========================================================================
# The fake model client
# ===========================================================================
class RateLimitError(Exception):
    """Stands in for an HTTP 429 'Too Many Requests' from a model provider."""


class FakeLLM:
    """A pretend LLM API with latency, a concurrency cap, and flaky failures."""

    def __init__(self, server_capacity: int = 5, failure_rate: float = 0.15):
        # server_capacity: max concurrent requests the "provider" accepts.
        #   Real providers enforce limits on concurrency, requests/minute
        #   and tokens/minute; we simulate the concurrency one.
        self.server_capacity = server_capacity
        # failure_rate: probability that a successful-looking call still fails
        #   with a transient network error (like a real flaky upstream).
        self.failure_rate = failure_rate
        # in_flight: how many requests are being processed right now.
        self.in_flight = 0
        # calls: total number of requests ever received (incl. retries).
        self.calls = 0

    async def complete(self, prompt: str) -> str:
        # `async def` makes this a COROUTINE FUNCTION. Calling it returns a
        # coroutine object; the body runs only when that object is awaited
        # or wrapped in a Task (see demo 2).
        self.calls += 1
        self.in_flight += 1
        try:
            # Too many concurrent requests -> reject quickly with a 429,
            # exactly like a provider enforcing your account's limit.
            if self.in_flight > self.server_capacity:
                await asyncio.sleep(0.05)  # a 429 still costs a round trip
                raise RateLimitError(f"429: {self.in_flight} concurrent requests")

            # asyncio.sleep is the async stand-in for "waiting on the network".
            # While this coroutine is suspended here, the event loop is free
            # to run OTHER coroutines. That is the entire point of async.
            # (time.sleep would freeze everything instead -- see demo 6.)
            await asyncio.sleep(random.uniform(0.3, 1.0))

            # Simulate a transient upstream failure (timeout, 502, reset...).
            if random.random() < self.failure_rate:
                raise ConnectionError("transient upstream error")

            return f"summary({prompt})"
        finally:
            # `finally` guarantees the counter is decremented whether the call
            # succeeded, raised, or was CANCELLED (cancellation is delivered as
            # an asyncio.CancelledError raised at the current `await`).
            self.in_flight -= 1


# ===========================================================================
# Demo 1. Why async at all: I/O-bound fan-out
# ===========================================================================
async def demo_sequential_vs_concurrent():
    # High capacity + no failures: isolate the pure latency effect.
    llm = FakeLLM(server_capacity=100, failure_rate=0)
    prompts = [f"doc-{i}" for i in range(10)]

    # --- Sequential: each await finishes before the next call starts.
    #     Total time ~= SUM of all latencies.
    t = time.perf_counter()
    for p in prompts:
        await llm.complete(p)
    seq = time.perf_counter() - t

    # --- Concurrent: asyncio.gather wraps every coroutine in a Task, starts
    #     them all, and waits for all of them. Total time ~= the SLOWEST call.
    #     Note: the generator expression `(llm.complete(p) for p in prompts)`
    #     is unpacked with * into separate arguments.
    t = time.perf_counter()
    await asyncio.gather(*(llm.complete(p) for p in prompts))
    conc = time.perf_counter() - t

    print(f"10 calls sequential: {seq:.2f}s | concurrent: {conc:.2f}s")
    # Takeaway: this is why embedding jobs, eval runs and multi-retriever RAG
    # are written async -- the CPU is idle while waiting on the model.


# ===========================================================================
# Demo 2. Coroutines are lazy; Tasks are scheduled immediately
# ===========================================================================
async def demo_coroutines_are_lazy():
    llm = FakeLLM(failure_rate=0)

    # Calling an async function only CREATES a coroutine object.
    # Nothing inside complete() has run yet -> calls stays 0.
    coro = llm.complete("x")
    await asyncio.sleep(0.3)  # give it plenty of time; still nothing happens
    print(f"coroutine created, 0.3s later: calls={llm.calls}")

    # create_task() hands the coroutine to the event loop, which will start
    # running it at the next opportunity (the next time we yield control).
    task = asyncio.create_task(llm.complete("y"))
    # sleep(0) = "yield control to the loop once". The loop starts the task,
    # which runs until its first internal await.
    await asyncio.sleep(0)
    print(f"create_task + one yield:      calls={llm.calls}")

    # Always await (or otherwise keep a reference to) tasks you create.
    # The loop holds only a WEAK reference to tasks, so an unreferenced
    # fire-and-forget task can be garbage-collected mid-flight.
    await task

    # Closing the never-awaited coroutine silences Python's
    # "RuntimeWarning: coroutine ... was never awaited".
    # EXPERIMENT: delete this line and read the warning -- in real code that
    # warning almost always means you forgot an `await`.
    coro.close()


# ===========================================================================
# Demo 3. Naive fan-out hits the provider's rate limit
# ===========================================================================
async def demo_unbounded_fanout():
    # The "provider" accepts only 5 concurrent requests.
    llm = FakeLLM(server_capacity=5, failure_rate=0)

    # Fire 20 at once. return_exceptions=True means gather does NOT raise on
    # the first error; instead each failed slot in the results list holds the
    # exception object. Use this for best-effort batches (eval runs, backfills).
    results = await asyncio.gather(
        *(llm.complete(f"doc-{i}") for i in range(20)),
        return_exceptions=True,
    )

    # Separate failures from successes by type.
    errors = [r for r in results if isinstance(r, Exception)]
    print(f"20 requests at once -> {len(errors)} failed "
          f"({type(errors[0]).__name__ if errors else 'none'})")
    # Takeaway: "make it concurrent" is easy; the real engineering is
    # controlling HOW concurrent (demo 4).


# ===========================================================================
# Demo 4. Production pattern: semaphore + timeout + backoff with jitter
# ===========================================================================
# Only retry errors that might succeed on a second try. Never retry e.g. a
# 400 Bad Request or an auth error -- retrying those just burns time/money.
RETRYABLE = (RateLimitError, ConnectionError, TimeoutError)


async def call_with_retry(llm: FakeLLM, prompt: str, sem: asyncio.Semaphore,
                          *, max_attempts: int = 4, timeout_s: float = 2.0) -> str:
    # The `*` makes max_attempts/timeout_s keyword-only arguments, so call
    # sites read clearly: call_with_retry(llm, p, sem, max_attempts=6).
    for attempt in range(1, max_attempts + 1):
        try:
            # Semaphore(N): at most N coroutines can be inside this block at
            # once; the rest wait at `async with sem`. This is your client-side
            # concurrency cap, set at or below the provider's limit.
            async with sem:
                # asyncio.timeout (3.11+): if the body takes longer than
                # timeout_s, the inner await is cancelled and TimeoutError is
                # raised. Never let a model call hang forever.
                async with asyncio.timeout(timeout_s):
                    return await llm.complete(prompt)
        except RETRYABLE:
            # Out of attempts -> let the error propagate to the caller.
            if attempt == max_attempts:
                raise
            # Exponential backoff: 0.25s, 0.5s, 1s, 2s ... capped at 8s.
            # Jitter (random 0.5x-1.5x) spreads retries out so many clients
            # that failed together don't all retry at the same instant
            # (the "thundering herd" problem).
            backoff = min(8.0, 0.25 * 2 ** (attempt - 1)) * random.uniform(0.5, 1.5)
            # IMPORTANT: this sleep is OUTSIDE `async with sem`, so a waiting
            # coroutine does not hold a concurrency slot while doing nothing.
            # EXPERIMENT: indent this sleep inside the semaphore block and
            # compare total time.
            await asyncio.sleep(backoff)
    raise RuntimeError("unreachable")  # keeps type-checkers happy


async def demo_bounded_with_retries():
    # Realistic provider: concurrency cap of 5 AND 15% transient failures.
    llm = FakeLLM(server_capacity=5, failure_rate=0.15)
    # Client-side cap matching the provider's limit.
    # EXPERIMENT: try Semaphore(10) and watch API calls (and 429s) go up.
    sem = asyncio.Semaphore(5)
    prompts = [f"doc-{i}" for i in range(30)]

    t = time.perf_counter()
    results = await asyncio.gather(
        *(call_with_retry(llm, p, sem) for p in prompts), return_exceptions=True
    )
    ok = sum(not isinstance(r, Exception) for r in results)
    # llm.calls > 30 because retried attempts count as extra calls.
    print(f"{ok}/30 succeeded | {llm.calls} API calls incl. retries | "
          f"{time.perf_counter() - t:.2f}s")
    # NOTE for real SDKs: AsyncOpenAI / AsyncAnthropic already retry 429/5xx
    # with backoff (configurable via max_retries). Don't stack your own retry
    # loop on top without disabling theirs, or retries multiply.


# ===========================================================================
# Demo 5. Structured concurrency: TaskGroup cancels siblings on failure
# ===========================================================================
async def demo_taskgroup():
    llm = FakeLLM(server_capacity=100, failure_rate=0)

    # Simulates a parallel safety/guardrail check that rejects the request.
    async def guardrail_check():
        await asyncio.sleep(0.2)
        raise ValueError("guardrail check failed")

    try:
        # TaskGroup (3.11+): every task created inside the `async with` block
        # is owned by the group. The block exits only when ALL tasks finish.
        # If ANY task raises, the group cancels all remaining siblings, then
        # re-raises the error(s) bundled in an ExceptionGroup.
        async with asyncio.TaskGroup() as tg:
            t1 = tg.create_task(llm.complete("retrieve"))
            t2 = tg.create_task(llm.complete("rerank"))
            tg.create_task(guardrail_check())
    # `except*` (3.11+) matches exceptions INSIDE an ExceptionGroup.
    # eg.exceptions is the tuple of matching ValueErrors.
    except* ValueError as eg:
        print(f"caught {eg.exceptions[0]!r}; "
              f"siblings cancelled: {t1.cancelled()}, {t2.cancelled()}")
    # Takeaway: when one step fails and the others' results are useless,
    # TaskGroup stops you paying for (and waiting on) wasted LLM calls.
    # Use gather(return_exceptions=True) when you want partial results.


# ===========================================================================
# Demo 6. Blocking the event loop (the #1 async bug in AI services)
# ===========================================================================
async def heartbeat(stop: asyncio.Event) -> float:
    """Ticks every 50 ms and returns the longest gap between ticks.

    If the loop is healthy, every gap is ~0.05s. If something blocks the loop,
    the heartbeat can't run, and the gap grows to the length of the block.
    This is essentially how loop-lag monitoring works in production.
    """
    worst, last = 0.0, time.perf_counter()
    while not stop.is_set():          # asyncio.Event = a simple async flag
        await asyncio.sleep(0.05)
        now = time.perf_counter()
        worst, last = max(worst, now - last), now
    return worst


def sync_work():
    # A plain (non-async) blocking call. Stands in for: a sync SDK client
    # (OpenAI() instead of AsyncOpenAI()), requests.get, a sync DB driver,
    # reading a big file, or heavy tokenisation/parsing.
    time.sleep(1.0)


async def demo_blocking_the_loop():
    for label in ("direct sync call", "asyncio.to_thread"):
        stop = asyncio.Event()
        hb = asyncio.create_task(heartbeat(stop))   # start the monitor
        await asyncio.sleep(0.1)                    # let it tick a little
        if label == "direct sync call":
            # BAD: runs on the event-loop thread. For 1s NOTHING else on this
            # loop can run -- in a FastAPI server, every user's request stalls.
            sync_work()
        else:
            # GOOD: to_thread runs the function in a worker thread and gives
            # back an awaitable; the loop keeps serving other coroutines.
            # (For CPU-heavy pure-Python work, use a ProcessPoolExecutor
            #  instead -- the GIL limits threads for CPU-bound code.)
            await asyncio.to_thread(sync_work)
        stop.set()                                  # stop the heartbeat
        print(f"{label:18} -> longest loop stall {await hb:.2f}s")


# ===========================================================================
# Demo 7. Handle results as they finish (progress bars, early output)
# ===========================================================================
async def demo_as_completed():
    llm = FakeLLM(server_capacity=100, failure_rate=0)
    # Start all tasks up front.
    tasks = [asyncio.create_task(llm.complete(f"doc-{i}")) for i in range(5)]
    # as_completed yields awaitables in COMPLETION order, not submission
    # order -- so fast results can be shown/logged/saved immediately.
    for i, fut in enumerate(asyncio.as_completed(tasks), 1):
        print(f"  [{i}/5] {await fut}")
    # Use for: eval-run progress bars, writing results to disk as they land,
    # returning the first good answer from several models.


# ===========================================================================
# Entry point
# ===========================================================================
async def main():
    # Functions are objects, so we can keep them in a list and loop over them.
    demos = [
        demo_sequential_vs_concurrent, demo_coroutines_are_lazy,
        demo_unbounded_fanout, demo_bounded_with_retries, demo_taskgroup,
        demo_blocking_the_loop, demo_as_completed,
    ]
    for n, demo in enumerate(demos, 1):
        print(f"\n=== {n}. {demo.__name__} ===")
        await demo()


if __name__ == "__main__":
    # asyncio.run creates an event loop, runs main() to completion, then
    # closes the loop. Call it ONCE, at the program's top level. (Inside
    # Jupyter or FastAPI a loop already exists -- there you just `await`.)
    asyncio.run(main())
