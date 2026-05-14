# Strategy: 4h_Camarilla_R1S1_Breakout_12hTrend_VolumeSpike_Rev2

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.428 | +36.5% | -7.2% | 284 | KEEP |
| ETHUSDT | 0.400 | +37.3% | -9.8% | 261 | KEEP |
| SOLUSDT | 0.287 | +37.3% | -17.4% | 225 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.347 | -3.5% | -5.6% | 113 | DISCARD |
| ETHUSDT | 1.042 | +18.9% | -5.2% | 101 | KEEP |
| SOLUSDT | 0.565 | +12.3% | -5.5% | 82 | KEEP |

## Code
```python
#!/usr/bin/env python3
name = "4h_Camarilla_R1S1_Breakout_12hTrend_VolumeSpike_Rev2"
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
    
    # === 12H DATA FOR TREND FILTER (EMA50) ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # === 12H DATA FOR CAMARILLA LEVELS ===
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    rng_12h = high_12h - low_12h
    R1_12h = close_12h + rng_12h * 1.1 / 12
    S1_12h = close_12h - rng_12h * 1.1 / 12
    R1_4h = align_htf_to_ltf(prices, df_12h, R1_12h)
    S1_4h = align_htf_to_ltf(prices, df_12h, S1_12h)
    
    # === VOLUME SPIKE (20-period) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(R1_4h[i]) or
            np.isnan(S1_4h[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: PRICE BREAKS ABOVE R1 + ABOVE 12H EMA50 + VOLUME SPIKE
            if (close[i] > R1_4h[i] and 
                close[i] > ema50_12h_aligned[i] and
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: PRICE BREAKS BELOW S1 + BELOW 12H EMA50 + VOLUME SPIKE
            elif (close[i] < S1_4h[i] and 
                  close[i] < ema50_12h_aligned[i] and
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: PRICE BREAKS BELOW S1 (REVERSAL) OR BELOW 12H EMA50
            if close[i] < S1_4h[i] or close[i] < ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: PRICE BREAKS ABOVE R1 (REVERSAL) OR ABOVE 12H EMA50
            if close[i] > R1_4h[i] or close[i] > ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-12 05:54
