# Strategy: 12h_Donchian20_Breakout_1dTrend_VolumeSpike

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.051 | +22.6% | -10.7% | 24 | KEEP |
| ETHUSDT | -0.667 | -3.8% | -23.2% | 19 | DISCARD |
| SOLUSDT | 1.015 | +148.2% | -16.3% | 20 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.355 | +9.7% | -4.5% | 8 | KEEP |
| SOLUSDT | -0.037 | +4.4% | -8.7% | 5 | DISCARD |

## Code
```python
#!/usr/bin/env python3
# 12h_Donchian20_Breakout_1dTrend_VolumeSpike
# Hypothesis: 12-hour Donchian channel (20-period) breakouts capture medium-term trends.
# Combined with 1-day EMA trend filter and volume spikes to avoid false breakouts.
# Works in bull markets (long on upper band breakout + uptrend) and bear markets (short on lower band breakout + downtrend).
# Volume spike filters low-conviction moves. Target: 15-25 trades/year per symbol.

name = "12h_Donchian20_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 12h Donchian channel (20-period)
    high_ma = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate EMA34 for trend filter (daily)
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume spike detection: 2.5x average volume (10-period = ~5 days on 12h chart)
    vol_ma = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34, 10)  # Ensure we have Donchian, EMA34, and volume MA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(high_ma[i]) or np.isnan(low_ma[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above upper Donchian band, price above EMA34 (uptrend), volume spike
            if (high[i] > high_ma[i-1] and 
                close[i] > ema34_1d_aligned[i] and 
                volume[i] > 2.5 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian band, price below EMA34 (downtrend), volume spike
            elif (low[i] < low_ma[i-1] and 
                  close[i] < ema34_1d_aligned[i] and 
                  volume[i] > 2.5 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price breaks below lower Donchian band OR price crosses below EMA34
            if (low[i] < low_ma[i-1] or close[i] < ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price breaks above upper Donchian band OR price crosses above EMA34
            if (high[i] > high_ma[i-1] or close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-12 18:08
