# Strategy: 4h_Camarilla_H4L4_Breakout_12hEMA50_VolumeSpike

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.089 | +24.1% | -8.8% | 191 | PASS |
| ETHUSDT | 0.400 | +43.6% | -10.9% | 179 | PASS |
| SOLUSDT | 0.352 | +46.1% | -21.8% | 148 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.844 | -2.3% | -7.1% | 65 | FAIL |
| ETHUSDT | 0.580 | +15.0% | -8.6% | 61 | PASS |
| SOLUSDT | -0.001 | +5.3% | -14.2% | 48 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla H4/L4 breakout with 12h EMA50 trend filter and volume spike.
- Uses 12h HTF for trend alignment (more responsive than 1d, less noisy than 4h)
- Camarilla H4/L4 from prior 12h for structure
- Long: price breaks above H4 + volume > 2.0x 20-period avg + price > 12h EMA50
- Short: price breaks below L4 + volume > 2.0x 20-period avg + price < 12h EMA50
- Exit: price re-enters Camarilla H4-L4 range OR 12h EMA50 trend flip
- Discrete position sizing: ±0.25 to minimize fee churn
- Target: 75-200 total trades over 4 years (19-50/year) on 4h timeframe
- Works in bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume confirmation: > 2.0x 20-period average (tight to avoid overtrading)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 12h EMA50 for trend filter (HTF = 12h as specified)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 12h Camarilla levels (based on prior 12h OHLC)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h_prev = df_12h['close'].values
    
    # Camarilla formula: range = high - low
    # H4 = close + (high - low) * 1.1/2
    # L4 = close - (high - low) * 1.1/2
    rng = high_12h - low_12h
    camarilla_h4 = close_12h_prev + rng * (1.1 / 2)
    camarilla_l4 = close_12h_prev - rng * (1.1 / 2)
    
    # Align Camarilla levels to 4h timeframe
    h4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h4)
    l4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l4)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need 50 for EMA50, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(h4_aligned[i]) or
            np.isnan(l4_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.0x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above H4 + volume confirmation + price > 12h EMA50
            if (close[i] > h4_aligned[i] and 
                volume_confirm and 
                close[i] > ema_50_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below L4 + volume confirmation + price < 12h EMA50
            elif (close[i] < l4_aligned[i] and 
                  volume_confirm and 
                  close[i] < ema_50_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price re-enters below L4 (mean reversion) OR price < 12h EMA50 (trend flip)
            if close[i] < l4_aligned[i] or close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price re-enters above H4 (mean reversion) OR price > 12h EMA50 (trend flip)
            if close[i] > h4_aligned[i] or close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H4L4_Breakout_12hEMA50_VolumeSpike"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-23 22:14
