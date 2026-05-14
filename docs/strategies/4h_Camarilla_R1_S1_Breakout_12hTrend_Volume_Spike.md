# Strategy: 4h_Camarilla_R1_S1_Breakout_12hTrend_Volume_Spike

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.294 | +8.2% | -10.2% | 168 | FAIL |
| ETHUSDT | 0.173 | +28.4% | -11.0% | 150 | PASS |
| SOLUSDT | 0.332 | +43.2% | -22.6% | 114 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.278 | +9.5% | -9.3% | 53 | PASS |
| SOLUSDT | -0.113 | +3.8% | -7.4% | 41 | FAIL |

## Code
```python
#!/usr/bin/env python3
name = "4h_Camarilla_R1_S1_Breakout_12hTrend_Volume_Spike"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h trend: EMA50
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # 4h Camarilla R1/S1
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    pivot_4h = (high_4h + low_4h + close_4h) / 3
    range_4h = high_4h - low_4h
    r1_4h = close_4h + (range_4h * 1.0833)
    s1_4h = close_4h - (range_4h * 1.0833)
    r1_4h_aligned = align_htf_to_ltf(prices, df_4h, r1_4h)
    s1_4h_aligned = align_htf_to_ltf(prices, df_4h, s1_4h)
    
    # Volume spike: current volume > 2.0 x 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * volume_ma20
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        if np.isnan(r1_4h_aligned[i]) or np.isnan(s1_4h_aligned[i]) or np.isnan(ema_50_12h_aligned[i]) or np.isnan(volume_ma20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Close breaks above R1, above 12h EMA50, volume spike
            if close[i] > r1_4h_aligned[i] and close[i] > ema_50_12h_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below S1, below 12h EMA50, volume spike
            elif close[i] < s1_4h_aligned[i] and close[i] < ema_50_12h_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Close below S1 or below 12h EMA50
            if close[i] < s1_4h_aligned[i] or close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Close above R1 or above 12h EMA50
            if close[i] > r1_4h_aligned[i] or close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-11 21:33
