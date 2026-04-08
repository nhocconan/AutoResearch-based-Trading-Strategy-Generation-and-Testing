# Auto Research Review

Generated: 2026-04-08 00:02:13 UTC

# Research Loop Operational Review

## 1. Executive Summary

The system exhibits a **78.8% attrition rate** with critical inefficiencies in the generation-to-validation pipeline. The dominant failure modes—`too_few_trades` (90) and `negative_sharpe` (93)—account for 76% of all train failures, indicating the generator produces strategies with entry conditions that are either too restrictive or fundamentally misaligned with market dynamics. The 21.2% keep rate represents significant compute waste that can be addressed through structural changes to the research loop.

## 2. Structural Weaknesses

**A. No Feedback Loop from Failures to Generator**
`agent_research.py` appears to generate strategies without ingesting failure pattern data. The `too_few_trades: 90` count suggests repeated generation of over-filtered entries that could be pre-emptively caught.

**B. Asymmetric Failure Distribution**
- Overtrading: 5 failures (generator too conservative on trade frequency)
- Too few trades: 90 failures (entry filters too restrictive)
This 18:1 ratio indicates the generator over-corrects for overtrading warnings in `program.md`.

**C. Static Knowledge Base Staleness**
`program.md` contains hard-coded timeframe keep rates from "16,000+ experiments" but no mechanism exists to update these priors from recent results. The "unknown" category in keep rates suggests tracking metadata gaps.

**D. Invalid Code Generation**
The "other" bucket (50 failures) likely contains syntax errors, invalid MTF patterns, or constraint violations that static checks should catch pre-backtest.

## 3. Highest-Impact Changes

| Priority | Change | Expected Impact |
|----------|--------|-----------------|
| 1 | Pre-backtest trade count estimator | Reduce `too_few_trades` by 60%+ |
| 2 | Failure pattern injection into prompts | Reduce `negative_sharpe` by 40% |
| 3 | Two-stage validation (syntax → semantic → backtest) | Eliminate "other" bucket |
| 4 | Dynamic `program.md` updates from results | Improve keep rate to 35%+ |

## 4. Model Split Recommendation

| Role | Model | Justification |
|------|-------|---------------|
| **Generation** | `deepseek-v3.2` | Strong code synthesis, follows structured constraints well, cost-effective for high-volume iteration |
| **Analysis/Review** | `kimi-k2-thinking` | Explicit reasoning chain for failure pattern recognition, better at identifying *why* strategies fail |

## 5. Next 7 Concrete Actions

1. **Add trade-count pre-filter in `agent_research.py`**: Before backtest, estimate expected trades using signal frequency on sample data. Reject if <5 trades/year estimated.

2. **Create `failure_patterns.md`**: Auto-generate weekly from `results.tsv` failures. Inject into generator prompt as "strategies that failed with reasons."

3. **Implement entry-condition relaxation heuristic**: When `too_few_trades` detected, auto-generate variant with one filter removed. Run as parallel experiment.

4. **Add per-symbol trade count validation**: Current rules allow 5 total trades; enforce minimum per-symbol (e.g., ≥2 per symbol) to avoid concentration risk.

5. **Fix "unknown" timeframe tracking**: Audit `auto_concept_research.py` to ensure TF metadata propagates to results logging.

6. **Create `validate_syntax.py` pre-check**: Run AST-based validation for MTF rules, lookahead patterns, and forbidden constructs before backtest dispatch.

7. **Weekly `program.md` prior update**: Script to recalculate timeframe keep rates from last 500 experiments and update the markdown table automatically.

---

**Word count: 498**
