# Sub-Agent Worker Contract

Every conversion sub-agent must follow this contract exactly.

## Scope

- Own only the assigned queue shard.
- Keep all files inside `tradingview-strategies/`.
- Do not edit repo data or code outside `tradingview-strategies/`.

## Required Skill Discipline

- Reread `tradingview-strategies/pinescript-to-python-skill/SKILL.md` before starting.
- Apply `common-failure-constraints.md` on every conversion and repair pass.
- Apply `mapping-checklist.md` before saving any strategy.
- Apply `testing-contract.md` before any suite backtest or dashboard update.
- Respect the repo contract and next-bar execution semantics.

## Model Requirement

- Use the repo `.env` official Ollama setup with `OLLAMA_MODEL` or `OLLAMA_CONVERT_MODEL` for classify, convert, and repair.
- Do not silently switch to another model inside the pipeline worker.

## Conversion Rules

- Classify first: `direct`, `partial`, or `unsupported`.
- If unsupported, write the report and do not fake parity.
- If import fails, repair against the exact import error.
- If signal generation on local repo data fails, repair against the exact runtime error.
- If backtest fails, repair against the exact runtime/backtest error.
- Do not introduce lookahead or same-bar fills.
- If the worker finds an obvious pipeline or extractor bug, fix it immediately and persist the repair before continuing the queue. Do not bounce that decision back to the user.
- Retry any single conversion at most 3 times, then persist the failure and continue the queue.
- If the assigned script already exhausted the Qwen batch retries, switch that specific repair task to Codex instead of burning more Qwen attempts.

## Output Discipline

- Save raw Pine under `tradingview-strategies/raw-pine/`.
- Save Python strategy modules under `tradingview-strategies/python-strategies/`.
- Save JSON classification reports under `tradingview-strategies/results/bulk/`.
- Save markdown reports under `tradingview-strategies/reports/bulk/`.
- Save worker-local state, manifest, log, and backtest shards under worker-specific filenames.
- Keep fair-comparison TradingView backtests on the shared fixed size of `$1,000` on `$10,000` initial capital.
- Keep Binance funding enabled unless the task explicitly requests a separate parity-check run.

## Browser Discipline

- If browser automation is needed, use named `agent-browser` sessions only.
- Close the `agent-browser` session immediately after each extraction task.
- Do not leave `Google Chrome for Testing` or `agent-browser` daemons running once the browser work for that batch is finished.
- Stage 2 conversion workers must not start browser automation at all; they only consume local Pine cache from Stage 1.

## Crawl Discipline

- For TradingView listing crawl work, process at most 20 pages per batch.
- Finish and persist one 20-page batch before starting the next 20-page batch.

## Completion

- After running a worker batch, merge shard outputs into the canonical dashboard files.
- Report the worker id, shard ownership, and the latest merged counts.
