# Strategy: 1h_4h1d_trend_volume_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.136 | +14.3% | -9.9% | 546 | FAIL |
| ETHUSDT | 0.025 | +20.3% | -10.7% | 514 | PASS |
| SOLUSDT | 0.674 | +87.6% | -25.5% | 442 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.399 | +11.4% | -7.9% | 167 | PASS |
| SOLUSDT | -0.317 | +0.4% | -12.5% | 164 | FAIL |

## Code
```python
#!/usr/bin/env python3
# 1h_4h1d_trend_volume_v1
# Hypothesis: Uses 4-hour and 1-day EMA trend filters with 1-hour volume-confirmed breakouts.
# Enters long when 1h price breaks above 4h EMA20 with volume spike and 1d uptrend.
# Enters short when 1h price breaks below 4h EMA20 with volume spike and 1d downtrend.
# Exits on opposite break or trend failure. Designed for 15-30 trades/year to avoid fee drag.
# Uses 1d trend filter for multi-timeframe alignment and 4h EMA for intermediate trend.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h1d_trend_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4-hour data for intermediate trend
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # 1-day data for long-term trend
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 4h EMA20 for intermediate trend
    ema20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    
    # 1d EMA50 for long-term trend
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 1-hour volume average (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if np.isnan(ema20_4h_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or np.isnan(avg_volume[i]):
            if position != 0:
                pass
            else:
                signals[i] = 0.0
            continue
        
        # Trend filters
        intermediate_uptrend = close[i] > ema20_4h_aligned[i]
        intermediate_downtrend = close[i] < ema20_4h_aligned[i]
        longterm_uptrend = close[i] > ema50_1d_aligned[i]
        longterm_downtrend = close[i] < ema50_1d_aligned[i]
        
        # Volume confirmation
        volume_ok = volume[i] > 2.0 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit: breakdown below 4h EMA or long-term trend failure
            if close[i] < ema20_4h_aligned[i] or not longterm_uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: break above 4h EMA or long-term trend failure
            if close[i] > ema20_4h_aligned[i] or not longterm_downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            if volume_ok:
                # Long entry: above 4h EMA with volume spike and long-term uptrend
                if intermediate_uptrend and longterm_uptrend:
                    position = 1
                    signals[i] = 0.20
                # Short entry: below 4h EMA with volume spike and long-term downtrend
                elif intermediate_downtrend and longterm_downtrend:
                    position = -1
                    signals[i] = -0.20
    
    return signals
```

## Last Updated
2026-04-08 16:45
