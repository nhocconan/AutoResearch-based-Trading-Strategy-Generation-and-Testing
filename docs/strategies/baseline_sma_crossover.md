# Strategy: SMA Crossover Baseline

## Hypothesis
Simple moving average crossover should capture medium-term trends in crypto futures.
This serves as the baseline to beat - any viable strategy must outperform this.

## Logic
- Long when 20-period SMA > 50-period SMA
- Short when 20-period SMA < 50-period SMA
- Flat during warmup period (first 50 bars)

## Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| fast_period | 20 | Fast SMA lookback |
| slow_period | 50 | Slow SMA lookback |
| timeframe | 1h | Operating timeframe |
| leverage | 1x | No leverage |

## Results (Train: 2021-2024)
*To be filled after first run*

## Results (Test: 2025+)
*To be filled after evaluation*

## Observations
- This is a trend-following strategy
- Expected to perform well in trending markets, poorly in ranging markets
- High transaction costs from frequent position flips
- No risk management (always fully positioned)

## Status
Baseline - reference strategy for comparison

## Last Updated
2026-03-20
