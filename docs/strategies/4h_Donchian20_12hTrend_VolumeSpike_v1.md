# Strategy: 4h_Donchian20_12hTrend_VolumeSpike_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.044 | +21.3% | -18.5% | 114 | PASS |
| ETHUSDT | 0.295 | +39.4% | -14.2% | 107 | PASS |
| SOLUSDT | 0.977 | +176.8% | -30.0% | 104 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.316 | -8.2% | -11.1% | 49 | FAIL |
| ETHUSDT | 0.047 | +5.8% | -11.1% | 40 | PASS |
| SOLUSDT | 0.495 | +15.2% | -11.8% | 37 | PASS |

## Code
```python
#!/usr/bin/env python3
name = "4h_Donchian20_12hTrend_VolumeSpike_v1"
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
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter (as per experiment: HTF = 12h)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA50 trend filter
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Donchian(20) breakout
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume filter: current volume > 1.8 * 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_ok = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(21, 31)  # 20 for Donchian + 1 shift, 30 for volume MA + 1
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: breakout above Donchian high + above 12h EMA50 + volume spike
            if close[i] > high_20[i] and close[i] > ema_50_12h_aligned[i] and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # Short: breakdown below Donchian low + below 12h EMA50 + volume spike
            elif close[i] < low_20[i] and close[i] < ema_50_12h_aligned[i] and volume_ok[i]:
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: price returns to Donchian range or breaks in opposite direction
            if position == 1:
                if close[i] < low_20[i] or close[i] < ema_50_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > high_20[i] or close[i] > ema_50_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-07 05:46
