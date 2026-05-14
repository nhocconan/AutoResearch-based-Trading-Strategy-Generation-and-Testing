# Strategy: 4h_12hCrossover_1dTrend_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.141 | +27.1% | -15.3% | 133 | PASS |
| ETHUSDT | 0.204 | +32.5% | -19.5% | 145 | PASS |
| SOLUSDT | 0.817 | +159.3% | -31.3% | 173 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.384 | +0.5% | -6.1% | 52 | FAIL |
| ETHUSDT | 0.437 | +14.1% | -9.2% | 45 | PASS |
| SOLUSDT | 0.349 | +12.7% | -10.0% | 48 | PASS |

## Code
```python
#!/usr/bin/env python3
name = "4h_12hCrossover_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h EMA crossover for trend direction
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    ema_12h_fast = pd.Series(close_12h).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema_12h_slow = pd.Series(close_12h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_12h_fast_aligned = align_htf_to_ltf(prices, df_12h, ema_12h_fast)
    ema_12h_slow_aligned = align_htf_to_ltf(prices, df_12h, ema_12h_slow)
    
    # 1d trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > vol_ma
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        if np.isnan(ema_12h_fast_aligned[i]) or np.isnan(ema_12h_slow_aligned[i]) or np.isnan(ema_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: 12h EMA cross up + above 1d EMA + volume
            if ema_12h_fast_aligned[i] > ema_12h_slow_aligned[i] and close[i] > ema_1d_aligned[i] and vol_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: 12h EMA cross down + below 1d EMA + volume
            elif ema_12h_fast_aligned[i] < ema_12h_slow_aligned[i] and close[i] < ema_1d_aligned[i] and vol_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: 12h EMA cross down or below 1d EMA
            if ema_12h_fast_aligned[i] < ema_12h_slow_aligned[i] or close[i] < ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: 12h EMA cross up or above 1d EMA
            if ema_12h_fast_aligned[i] > ema_12h_slow_aligned[i] or close[i] > ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-11 21:51
