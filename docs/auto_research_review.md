# Auto Research Review

Generated: 2026-05-14 02:32:48 UTC

# Research-Ops Review: Crypto Strategy Autoresearch Loop

## 1. Executive Summary
The research loop is suffering a severe sample efficiency crisis. 90 out of 240 recent candidates failed for `too_few_trades` and 85 for `negative_sharpe`, while only 4 failed for `overtrading`. The system is generating structurally broken or overly restrictive code 73% of the time, resulting in a dismal 25.4% keep rate. The top test winners (e.g., `1h_SwingReversal_4hTrend_Filter` with Sharpe >9) prove viable alpha exists, but the generator lacks the failure-feedback mechanisms to navigate the parameter space efficiently. We must pivot from blind generation to targeted refinement.

## 2. Structural Weaknesses
- **Failure Asymmetry:** The `too_few_trades` bucket (90) massively outweighs `overtrading` (4). `agent_research.py` is over-indexing on the "fee drag is #1 killer" heuristic, producing hyper-conservative strategies that never trigger entries.
- **Invalid Code / Negative Sharpe:** 85 `negative_sharpe` failures imply `auto_concept_research.py` is proposing concepts, or `agent_research.py` is implementing them, with fundamentally flawed logic or look-ahead bias that slips past static checks.
- **Lookahead Leakage:** The top `1h` winners have suspiciously high Sharpes (>9) and trade counts (241-305) on a timeframe with a historical 17% keep rate. The prefix look-ahead test is likely passing, but intrabar SL/TP simulation or data-leakage in `generate_signals()` may be artificially inflating results.
- **Flat Feedback Loop:** `agent_research.py` currently operates open-loop. It logs to `results.tsv` but doesn't programmatically parse its own failure modes to adjust the next prompt.

## 3. Highest-Impact Changes
- **Dynamic Trade-Count Targeting:** Hardcode `research_rules.py` thresholds into the `agent_research.py` prompt dynamically. If the last 5 runs failed for `too_few_trades`, force the model to loosen entry filters (e.g., drop a volume spike requirement).
- **Prefix-Test Bootstrapping:** Move the prefix look-ahead test *before* the full backtest in the evaluation flow. Running a 100-bar prefix check costs milliseconds; running a 4-year backtest costs minutes. Discard 85% of `negative_sharpe` / look-ahead-biased strategies before paying the compute cost of a full backtest.
- **Overtrading Circuit Breaker:** Add a post-generation static check in `agent_research.py`: if the strategy logic contains more than 3 independent entry conditions on a 1h/4h timeframe, reject the code block *before* backtesting.

## 4. Model Split Recommendation
**Maintain `glm-5.1:cloud` as the single model for both generation and review.** 
Splitting models (e.g., a smaller model for review, larger for generation) introduces serialization latency and context-format fragility that degrades autonomous loops. `glm-5.1:cloud` offers the necessary instruction-following rigor for structured code generation and the analytical capacity for self-review. Instead of splitting the model, split the *prompt context*: use `glm-5.1:cloud` with a strict two-pass profile (Generator mode vs. Critic mode) within `agent_research.py` to evaluate its own code before submitting to the backtest engine.

## 5. Next 7 Concrete Actions
1. **Patch `agent_research.py` failure injection:** Parse `results.tsv` at runtime. Inject the exact failure bucket (e.g., "Last 3 runs failed: too_few_trades. Reduce entry filters by 1") into the next LLM prompt.
2. **Add trade-count guard in `agent_research.py`:** Pre-parse the generated `strategy.py`. If it uses >3 entry filters for `<4h` timeframes, force a retry to combat `too_few_trades`.
3. **Reorder `evaluate.py` flow:** Execute the prefix look-ahead test *first*. Skip the full 4-year backtest if the prefix fails, saving ~80% of compute on `negative_sharpe` candidates.
4. **Audit `1h` winners:** Run the top 3 `1h_SwingReversal` strategies through a manual intrabar fill simulation. A 9.4 Sharpe on 1h is a statistical red flag for implicit look-ahead.
5. **Update `program.md` trade targets:** Explicitly map `too_few_trades` to the prompt: "If trades < 5, your entry condition is too strict. Remove one filter."
6. **Constrain `auto_concept_research.py`:** Limit concept generation to 4h, 12h, and 1d timeframes only. 1h and below are wasting compute given current keep rates.
7. **Implement Critic Pass in `agent_research.py`:** Before writing `strategy.py` to disk, ask `glm-5.1:cloud` to critique the generated code: "Does this strategy contain look-ahead bias? Will it generate >5 trades on 4h data?" Reject if confidence is low.
