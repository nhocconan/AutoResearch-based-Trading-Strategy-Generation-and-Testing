# Agent Rules

Operational pass/fail rules for this repo are centralized in [`research_rules.py`](research_rules.py).

Authoritative files:
- [`research_rules.py`](research_rules.py): train/test thresholds and DD limit
- [`program.md`](program.md): research protocol and evaluation flow
- [`CLAUDE.md`](CLAUDE.md): concise project rules for coding agents
- [`STRATEGY_RULES.md`](STRATEGY_RULES.md): strategy-writing constraints

Hard evaluation flow:
1. Validate strategy code
2. Run prefix look-ahead audit on the saved code before trusting any backtest result
2. For each symbol independently: run train
3. Only if that symbol passes train, run test for that same symbol
4. A strategy is kept only if at least one symbol passes both train and test
5. Rows with `status='discard'` are research history, not active kept results

Hard anti-lookahead rules:
1. `align_htf_to_ltf()` only makes a raw HTF bar available after that HTF bar closes
2. Any lagging HTF indicator that needs future HTF bars for confirmation must add explicit extra delay
3. Williams fractals must use `additional_delay_bars=2` when aligned from HTF to LTF
4. Previous-period pivots/regimes must be built from fully completed prior HTF periods only
5. Any strategy that fails static validation or the prefix look-ahead audit is invalid and must be purged from `strategies/`, `docs/strategies/`, and `results.db`

Hard LLM/runtime model rule:
1. Auto-research generation and analysis must use Ollama Cloud model `glm-5.1:cloud`
2. Do not silently fall back to a different model for auto-research coding/reasoning tasks

Hard self-improvement flow:
1. Every 12 hours, or sooner if stale inputs are detected, run `auto_concept_research.sh`
2. Order is fixed: `internet_strategy_discovery.py` -> `auto_process_review.py` -> `auto_concept_research.py` -> `verification_remediation.py`
3. The cycle must be single-instance only; use the shell lock in `auto_concept_research.sh`
4. `agent_research.py` may trigger the cycle when the discovery/review artifacts are stale; cron is not the only enforcement path
