## Goal
Implement:
- MCPConnectionManager in `src/mcp_client.py` with singleton lifecycle, async `start()`, `get_tools()`, `close()`, SSE heartbeat (30–60s), and exponential backoff reconnect.
- Unified async execution: route all async via `utils.run_async`; add cleanup hooks to close MCP client and cancel lingering tasks on reruns/shutdown.

### Plan 1: MCPConnectionManager

- Summary
  - Create a singleton manager around `MultiServerMCPClient` to own the connection, provide tools, keep the SSE alive with a heartbeat, and transparently reconnect using exponential backoff with jitter.
  - Integrate with Streamlit lifecycle via `st.session_state` and explicit cleanup.

- Steps
  1. Class scaffold
     - File: `src/mcp_client.py`
     - Add `MCPConnectionManager` with:
       - Fields: `client: MultiServerMCPClient | None`, `server_config: Dict[str, Dict]`, `lock: asyncio.Lock`, `heartbeat_task: asyncio.Task | None`, `reconnect_task: asyncio.Task | None`, `running: bool`, `tools_cache: List[BaseTool]`, `last_heartbeat_ok: bool`.
       - Singleton: expose `get_instance()` that stores one instance in a module-level variable; guard re-entry with `lock`.
  2. Async API
     - `async start(server_config: Dict[str, Dict]) -> None`
       - If already running and config unchanged, no-op.
       - If running but config changed, `await close()` then re-init.
       - Instantiate `MultiServerMCPClient(server_config)`; consider a first `await client.get_tools()` to validate connectivity early.
       - Create heartbeat task: `self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())`.
     - `async get_tools(force_refresh: bool = False) -> List[BaseTool]`
       - If no client: raise or return empty; optionally auto-start if config exists.
       - If cache empty or `force_refresh`, fetch `await client.get_tools()`; cache results.
       - Return `tools_cache`.
     - `async close() -> None`
       - Set `running = False`.
       - Cancel `heartbeat_task` and `reconnect_task` (await cancellation and swallow `CancelledError`).
       - If client has `aclose()` or close semantics, await it; then clear references and caches.
  3. Heartbeat loop (SSE keepalive)
     - Implement `_heartbeat_loop(interval_sec: int = 45)`
       - While `running`: `await asyncio.sleep(interval_sec)`.
       - Send a lightweight request that exercises the connection:
         - Preferred: a no-op/ping tool if your server exposes it.
         - Fallback: `await self.client.get_tools()` (be cautious with rate limits; 45–60s interval).
       - On success: `last_heartbeat_ok = True`.
       - On failure: set `last_heartbeat_ok = False`; schedule reconnect if not already scheduled.
     - Note: For SSE, servers often send comment frames to keep the stream open; client-side heartbeat is still useful when server doesn’t. See MDN notes on SSE heartbeats [Using server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events).
  4. Reconnect with backoff and jitter
     - Implement `_ensure_reconnect()` that idempotently starts `self.reconnect_task` if not running.
     - Implement `_reconnect_loop()`:
       - Capture `base` (e.g., 0.5s), `cap` (e.g., 30s).
       - Retry connect in a loop while `running`:
         - Backoff with full jitter: `sleep = min(cap, base * 2**attempt)` then `await asyncio.sleep(random.uniform(0, sleep))`.
         - Try to recreate `MultiServerMCPClient(self.server_config)` and validate (`await get_tools()`).
         - On success: swap in new client, refresh `tools_cache`, clear `reconnect_task`, break.
       - Reference: AWS Architecture Blog on backoff + jitter [Exponential Backoff And Jitter](https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/).
  5. Thread-safety and guards
     - Use `asyncio.Lock` to guard start/close, and when swapping client instance.
     - Ensure only one heartbeat and one reconnect task exist at a time.
  6. Integration points
     - When connecting servers (your UI code that calls `setup_mcp_client` today), replace with:
       - Acquire instance: `mgr = MCPConnectionManager.get_instance()`
       - `await mgr.start(server_config)` and `tools = await mgr.get_tools()`
       - Store `mgr` in `st.session_state.mcp_manager` for reuse.
     - Where tools are needed (agent creation), always fetch from the manager (do not instantiate new clients ad-hoc).
  7. Teardown
     - Before changing server config or on agent teardown, call `await mgr.close()`.
     - Add an atexit handler for safety to close outstanding connections during interpreter exit (in addition to Streamlit cleanup plan below).

- Notes and references
  - MCP adapters: `MultiServerMCPClient` usage and SSE timeouts are covered in the adapters repo and its examples: `https://github.com/langchain-ai/langchain-mcp-adapters`
  - SSE heartbeats (server comments): MDN [Using server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events)
  - Backoff with jitter: AWS blog [Exponential Backoff And Jitter](https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/)

### Plan 2: Unify async wrappers + cleanup hooks

- Summary
  - Use `utils.run_async` as the single entry for running async from Streamlit callbacks. Make other helpers thin aliases or remove them. Add a small registry to track background tasks for cancellation. Ensure MCP closes and tasks cancel on reruns/shutdown.

- Steps
  1. Consolidate async wrappers
     - Keep `utils.run_async` as the primary function. It already:
       - Accepts a factory or coroutine
       - Protects against “already awaited”
       - Uses a fresh loop/thread with timeouts
     - Deprecate direct usage of:
       - `run_streaming_async`
       - `run_async_coroutine`
       - `safe_async_call`
     - Implement them as thin passthroughs to `run_async` for backward compatibility, marking them internal or slated for removal:
       - `safe_async_call(coro_or_factory, error_message, timeout)` → `run_async(lambda: asyncio.wait_for(...))` with try/except around it for message formatting.
       - `run_streaming_async(async_gen_func)` → `run_async(lambda: run_async_generator(async_gen_func))`.
  2. Replace call sites
     - In `src/tab_components.py`:
       - Calls already using `run_async(process_streaming())` and `run_async(run_tool(...))` can stay.
       - Re-route any other direct async wrappers to `run_async`.
     - In `src/agent_manager.py`:
       - Keep `ainvoke`/`astream_events` usage inside async functions. Only the outermost calls from UI invoke `run_async`.
  3. Background task registry (optional but recommended)
     - Add a minimal registry in `utils`:
       - `register_task(name: str, task: asyncio.Task)` stores tasks in `st.session_state._bg_tasks`.
       - `cancel_all_tasks()` iterates tasks, cancels, awaits `CancelledError`.
     - Use it for long-lived tasks (e.g., MCP heartbeat/reconnect if not enclosed by the manager), otherwise rely on the MCP manager’s own cancellation.
  4. Cleanup hooks on rerun and shutdown
     - Rerun-safe init
       - At your app’s top-level (e.g., in `app.py` or first UI entry), before creating new connections:
         - If `st.session_state.get("mcp_manager")` exists, and new server config differs, `run_async(lambda: st.session_state.mcp_manager.close())`.
       - After that: create/get instance and `run_async(lambda: mgr.start(server_config))`.
     - Session end / shutdown
       - Streamlit doesn’t expose official per-session shutdown hooks; use two layers:
         - Explicit tear-down path when the user clicks “Disconnect” or changes provider/server: call `mgr.close()` and `cancel_all_tasks()`.
         - Global fallback: `atexit.register(...)` to attempt `mgr.close()` on process exit.
       - Consider `st.cache_resource` for resources that must be unique and cleaned up when the interpreter stops; see Streamlit docs for resource caching [st.cache_resource](https://docs.streamlit.io/develop/api-reference/caching/st.cache_resource). You still need explicit cleanup on rerun.
  5. Timeouts and cancellation
     - Expose a cancel UI control (e.g., “Cancel response”) that sets `st.session_state.cancel_requested = True`.
     - In long streaming loops (e.g., your `process_streaming` in `tab_components.py`), periodically check this flag and break out early, then clear the flag.
     - Ensure `run_async` propagates `TimeoutError` with clear messaging; use your existing `format_error_message` for display.
  6. Telemetry and logs
     - Log heartbeat successes/failures, reconnect attempts and durations, and task cancellations to make debugging easier.
     - In the UI, display connection status: Connected / Reconnecting / Disconnected, maybe with last heartbeat time.

- Notes and references
  - Streamlit resource lifecycle: `st.session_state` and rerun semantics; resource caching doc: [st.cache_resource](https://docs.streamlit.io/develop/api-reference/caching/st.cache_resource)
  - Async patterns and cancellation: Python asyncio docs on locks, tasks, cancellation [Python asyncio Locks](https://docs.python.org/3/library/asyncio-sync.html#locks)

### Acceptance criteria

- MCPConnectionManager
  - Only one instance per session; reconnects automatically with backoff (with jitter) after SSE failures.
  - Heartbeat runs at 30–60s interval; logs failure and triggers reconnect.
  - `get_tools()` returns cached tools and refreshes on demand; handles errors gracefully.
  - `close()` cancels tasks and releases the underlying client without leaving threads/loops behind.

- Unified async
  - All async calls from Streamlit routes go through `utils.run_async`.
  - No dangling tasks between reruns; MCP connection and tasks cleaned when reconfiguring or exiting.

If you want, I can implement the manager class and wire up the UI calls next.