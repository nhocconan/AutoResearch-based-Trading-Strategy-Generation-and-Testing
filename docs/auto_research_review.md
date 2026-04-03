# Auto Research Review

Generated: 2026-04-03 07:36:08 UTC

# Operational Review: Autonomous Crypto Strategy Research Loop

## 1. Executive Summary
The current research loop suffers from a **critical misalignment between generation priors and empirical success rates**. While the system correctly identifies high-yield timeframes (12h, 4h), the generator (`agent_research.py`) continues to produce strategies for dead zones (15m, 5m) and low-probability patterns (simple crossovers), resulting in a **~70% waste rate** on "too_few_trades" and "overtrading." The validation pipeline is reactive rather than preventive, allowing invalid or statistically weak code to consume backtest compute. Immediate optimization requires hard-coding empirical constraints into the generation prompt and splitting model roles to separate creative synthesis from rigorous auditing.

## 2. Structural Weaknesses
*   **Inefficient Sampling Distribution:** Despite `program.md` listing 15m/5m as "DEAD," the generator still allocates resources here. The 0% keep rate on these TFs indicates a failure to internalize historical failure data during the `auto_concept_research.py` phase.
*   **Reactive Validation:** Static checks and backtests occur *after* code generation. Strategies with obvious overtrading logic (e.g., missing regime filters on 4h) are generated, written to `strategy.py`, and only rejected after expensive backtesting.
*   **Weak Failure Feedback Loop:** The "too_few_trades" bucket (116 failures) suggests the generator does not understand the *mechanism* of trade frequency control (e.g., tightening entry thresholds vs. adding time-based cooldowns). It treats "low trades" as a random variance rather than a structural parameter tuning issue.
*   **Monolithic Model Role:** Using a single model pass for both concept invention and code implementation leads to "hallucinated compliance," where the model claims to follow `program.md` constraints (like MTF rules) but fails in implementation details, triggering static check failures.

## 3. Highest-Impact Changes
1.  **Pre-Generation Constraint Injection:** Modify `agent_research.py` to dynamically inject a "Negative Constraint Block" into the system prompt before every generation call. This block must explicitly list the "STOP PURSUING" items from `program.md` as forbidden tokens/logic paths, not just suggestions.
2.  **Synthetic Trade Count Estimator:** Before writing `strategy.py`, implement a lightweight logic check in `auto_concept_research.py` that estimates expected trade frequency based on the proposed logic (e.g., "Breakout on 4h without ADX filter" → Flag as "High Risk Overtrading"). Reject the concept *before* code gen.
3.  **Failure-Class Specific Fine-Tuning Prompts:** When `too_few_trades` occurs, the retry prompt must explicitly demand specific mechanisms (e.g., "Add volatility-based threshold scaling" or "Reduce lookback period") rather than generic "improve performance" instructions.
4.  **Strict MTF Template Enforcement:** Replace free-form MTF coding in `strategy.py` with a rigid scaffold. The generator should only fill the *signal logic* within a pre-validated template that guarantees `get_htf_data()` is called correctly, eliminating the "manual MTF" static check failures.

## 4. Model Split Recommendation
To maximize sample efficiency and code correctness, split the workflow between two distinct model personalities:

*   **Generator (Creative/Synthesis): `qwen3-next:80b`**
    *   *Justification:* Qwen demonstrates superior instruction following for complex coding tasks and mathematical logic synthesis. It excels at combining disparate indicators (e.g., "Pivot + Fisher + Regime") into coherent `strategy.py` code without syntax errors. Its large context window handles the full `program.md` constraints effectively.
*   **Reviewer (Critical/Analytical): `nemotron-3-super`**
    *   *Justification:* Nemotron is optimized for reasoning, critique, and identifying logical fallacies. It should act as the "Gatekeeper" *before* backtesting. Its role is to scan the generated `strategy.py` for lookahead bias, overtrading logic, and violation of the "STOP PURSUING" list. It provides the "Red Team" analysis that `qwen` ignores in its drive to create.

## 5. Next 7 Concrete Actions

1.  **Update `agent_research.py` Prompt:** Hard-code the "STOP PURSUING" list (15m, 5m, EMA crosses, Parabolic SAR) as a **negative constraint**. If the model proposes these, force an immediate regeneration.
2.  **Implement `pre_flight_check.py`:** Create a new script running between generation and backtesting. It parses `strategy.py` AST to count potential signal triggers on a sample of data. If estimated trades < 40 or > 500 (for 4h), reject immediately.
3.  **Refactor `strategy.py` Template:** Create a locked template file where the MTF data retrieval and loop structure are immutable. The generator only outputs the `calculate_signal()` function body. This eliminates 90% of static check failures.
4.  **Calibrate Trade Frequency Targets:** Update `program.md` and the generator's knowledge base to emphasize **"Total Trades over 4 Years"** (50-200 range) rather than annualized metrics. Explicitly instruct the model to calculate `expected_trades = (days * bars_per_day) * trigger_probability`.
5.  **Deploy Model Split:** Configure the pipeline to use `qwen3-next:80b` for `agent_research.py` and `nemotron-3-super` for a new `review_step.py` that runs before `backtest.py`.
6.  **Enhance Failure Feedback:** Modify the error handler in `evaluate.py`. If `too_few_trades`, append specific debugging hints to the context for the next run (e.g., "Your ADX threshold of 40 is too high; try 20-25").
7.  **Timeframe Re-weighting:** Adjust the `auto_concept_research.py` sampling weights to force **70% of new experiments** onto 12h and 4h timeframes, 20% on 1d, and strictly 10% on 6h/1h for novelty testing. Set 15m/5m/30m probability to 0.0.
