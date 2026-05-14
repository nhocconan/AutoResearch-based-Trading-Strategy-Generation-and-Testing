# Strategy: 6H_Pivot_R2_S2_Breakout_Volume_Volatility

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.126 | +26.0% | -11.8% | 143 | PASS |
| ETHUSDT | 0.068 | +22.1% | -13.6% | 140 | PASS |
| SOLUSDT | 0.424 | +59.2% | -22.3% | 109 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.902 | -11.2% | -13.9% | 45 | FAIL |
| ETHUSDT | 1.416 | +33.2% | -8.2% | 41 | PASS |
| SOLUSDT | -0.325 | -0.6% | -16.7% | 42 | FAIL |

## Code
```python
#!/usr/bin/env python3
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
    
    # Load 1d data for pivot points (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's pivot points (standard)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    prev_high = high_1d
    prev_low = low_1d
    prev_close = close_1d
    pivot = (prev_high + prev_low + prev_close) / 3
    r1 = 2 * pivot - prev_low
    s1 = 2 * pivot - prev_high
    r2 = pivot + (high_1d - low_1d)
    s2 = pivot - (high_1d - low_1d)
    
    # Align pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    
    # Volume confirmation: 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if data not ready
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(vol_avg_20[i]) or
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above R2 + volume spike + volatility filter
            if (close[i] > r2_aligned[i] and 
                volume[i] > 1.5 * vol_avg_20[i] and
                atr[i] > 0.5 * atr[i-1] if i > 0 else True):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S2 + volume spike + volatility filter
            elif (close[i] < s2_aligned[i] and 
                  volume[i] > 1.5 * vol_avg_20[i] and
                  atr[i] > 0.5 * atr[i-1] if i > 0 else True):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price crosses back to opposite pivot level (full exit)
            if position == 1:
                # Exit long: Price closes below S1
                if close[i] < s1_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: Price closes above R1
                if close[i] > r1_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6H_Pivot_R2_S2_Breakout_Volume_Volatility"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-22 11:41
