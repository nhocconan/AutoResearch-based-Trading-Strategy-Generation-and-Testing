# Strategy: 6h_elder_ray_1d_trend_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.209 | +8.5% | -12.7% | 293 | FAIL |
| ETHUSDT | 0.067 | +22.1% | -13.6% | 309 | PASS |
| SOLUSDT | 0.452 | +62.6% | -38.6% | 273 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.401 | +12.1% | -7.1% | 82 | PASS |
| SOLUSDT | 0.320 | +11.1% | -10.1% | 94 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Elder Ray Index + 1d Trend Filter
# Hypothesis: Elder Ray (bull/bear power) measures bull/bear strength relative to EMA.
# Combined with 1d EMA50 trend filter to avoid counter-trend trades.
# Works in both bull and bear markets by only taking trades aligned with higher timeframe trend.
# Targets 20-30 trades/year with disciplined entries to avoid overtrading.

name = "6h_elder_ray_1d_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False).mean().values
    ema50_6h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 6-day EMA for Elder Ray (13-period EMA on 6h chart)
    ema13 = pd.Series(close).ewm(span=13, adjust=False).mean().values
    
    # Elder Ray components
    bull_power = high - ema13  # Bull power: high - EMA
    bear_power = low - ema13   # Bear power: low - EMA
    
    # Smooth the power signals (6-period EMA)
    bull_power_smooth = pd.Series(bull_power).ewm(span=6, adjust=False).mean().values
    bear_power_smooth = pd.Series(bear_power).ewm(span=6, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup for EMAs
        # Skip if required data not available
        if (np.isnan(ema50_6h[i]) or 
            np.isnan(bull_power_smooth[i]) or 
            np.isnan(bear_power_smooth[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: bear power turns positive OR trend turns bearish
            if bear_power_smooth[i] > 0 or close[i] < ema50_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: bull power turns negative OR trend turns bullish
            if bull_power_smooth[i] < 0 or close[i] > ema50_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: strong bull power AND bearish trend alignment (bear power negative)
            if bull_power_smooth[i] > 0 and bear_power_smooth[i] < 0 and close[i] > ema50_6h[i]:
                position = 1
                signals[i] = 0.25
            # Short: strong bear power AND bullish trend alignment (bull power positive)
            elif bear_power_smooth[i] < 0 and bull_power_smooth[i] > 0 and close[i] < ema50_6h[i]:
                position = -1
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-07 14:01
