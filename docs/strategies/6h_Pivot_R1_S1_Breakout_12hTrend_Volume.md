# Strategy: 6h_Pivot_R1_S1_Breakout_12hTrend_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.237 | +30.7% | -7.9% | 155 | KEEP |
| ETHUSDT | 0.055 | +22.3% | -14.0% | 140 | KEEP |
| SOLUSDT | 0.315 | +42.1% | -14.4% | 117 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.221 | -5.6% | -10.5% | 55 | DISCARD |
| ETHUSDT | 0.960 | +22.1% | -7.5% | 49 | KEEP |
| SOLUSDT | 0.489 | +12.8% | -9.8% | 39 | KEEP |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Pivot_R1_S1_Breakout_12hTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h Pivot Points (R1, S1) once before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Classic pivot point calculation: P = (H+L+C)/3
    pivot_12h = (high_12h + low_12h + close_12h) / 3.0
    # R1 = 2*P - L, S1 = 2*P - H
    r1_12h = 2 * pivot_12h - low_12h
    s1_12h = 2 * pivot_12h - high_12h
    
    # Align Pivot levels to 6h timeframe (wait for 12h bar close)
    r1_12h_aligned = align_htf_to_ltf(prices, df_12h, r1_12h)
    s1_12h_aligned = align_htf_to_ltf(prices, df_12h, s1_12h)
    
    # 12h EMA50 for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Sufficient warmup for EMA50
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(r1_12h_aligned[i]) or np.isnan(s1_12h_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above R1 + uptrend + volume spike
            long_cond = (close[i] > r1_12h_aligned[i]) and \
                        (close[i] > ema_50_12h_aligned[i]) and \
                        volume_spike[i]
            # Short: break below S1 + downtrend + volume spike
            short_cond = (close[i] < s1_12h_aligned[i]) and \
                         (close[i] < ema_50_12h_aligned[i]) and \
                         volume_spike[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: close below S1 (mean reversion to support)
            if close[i] < s1_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: close above R1 (mean reversion to resistance)
            if close[i] > r1_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-08 04:41
