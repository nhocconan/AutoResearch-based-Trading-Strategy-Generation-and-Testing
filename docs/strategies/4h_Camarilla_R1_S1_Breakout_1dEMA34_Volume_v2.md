# Strategy: 4h_Camarilla_R1_S1_Breakout_1dEMA34_Volume_v2

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.086 | +23.8% | -12.9% | 173 | PASS |
| ETHUSDT | 0.117 | +25.4% | -14.6% | 164 | PASS |
| SOLUSDT | 0.958 | +157.2% | -22.7% | 146 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.896 | -3.7% | -8.6% | 64 | FAIL |
| ETHUSDT | 0.606 | +16.3% | -10.1% | 59 | PASS |
| SOLUSDT | 0.267 | +10.0% | -10.4% | 48 | PASS |

## Code
```python
#!/usr/bin/env python3
name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_Volume_v2"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA(34) for trend filter
    ema_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate Pivot Points from previous 1d
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    pivot = (prev_high + prev_low + prev_close) / 3
    r1 = 2 * pivot - prev_low
    s1 = 2 * pivot - prev_high
    
    # Align Pivot levels to 4h
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume filter: > 1.8x 20-period average (tightened from 1.5x)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > 1.8 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Wait for EMA and Pivots
    
    for i in range(start_idx, n):
        if np.isnan(ema_1d_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above R1 with daily uptrend and volume
            if (close[i] > r1_aligned[i] and close[i] > ema_1d_aligned[i] and vol_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: Break below S1 with daily downtrend and volume
            elif (close[i] < s1_aligned[i] and close[i] < ema_1d_aligned[i] and vol_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Close below S1 or trend change
            if close[i] < s1_aligned[i] or close[i] < ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Close above R1 or trend change
            if close[i] > r1_aligned[i] or close[i] > ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 4h Camarilla R1/S1 breakout with 1d EMA(34) trend filter and volume confirmation.
# Pivot levels from prior day provide key support/resistance. Breaking R1/S1 indicates momentum.
# Daily EMA filter ensures alignment with higher timeframe trend. Volume confirms institutional participation.
# Tightened volume filter (1.8x vs 1.5x) to reduce trade frequency and avoid overtrading.
# Target: 25-35 trades/year to minimize fee drift. Position size 0.25 limits drawdown in volatile markets.
```

## Last Updated
2026-05-07 14:28
