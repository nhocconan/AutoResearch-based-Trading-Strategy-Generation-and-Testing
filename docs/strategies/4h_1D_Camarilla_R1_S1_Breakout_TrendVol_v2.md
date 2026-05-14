# Strategy: 4h_1D_Camarilla_R1_S1_Breakout_TrendVol_v2

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.643 | +45.5% | -7.3% | 206 | PASS |
| ETHUSDT | 0.263 | +32.0% | -11.8% | 194 | PASS |
| SOLUSDT | 0.662 | +75.3% | -22.2% | 160 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.598 | +1.3% | -4.5% | 80 | FAIL |
| ETHUSDT | 0.780 | +16.5% | -7.3% | 76 | PASS |
| SOLUSDT | 0.411 | +11.1% | -10.1% | 55 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
4h_1D_Camarilla_R1_S1_Breakout_TrendVol_v2
Hypothesis: 4-hour breakouts from daily Camarilla R1/S1 levels with daily EMA50 trend filter and volume spike confirmation.
Only takes long when price breaks above R1 with volume spike and daily uptrend, short when breaks below S1 with volume spike and daily downtrend.
Uses tighter entry conditions (added ATR filter) to reduce trade frequency and improve robustness in both bull and bear markets.
"""

name = "4h_1D_Camarilla_R1_S1_Breakout_TrendVol_v2"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume spike: >2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # ATR for volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Daily data for Camarilla levels and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Daily EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Daily Camarilla R1 and S1 from previous day
    prev_close_1d = df_1d['close'].shift(1).values
    prev_high_1d = df_1d['high'].shift(1).values
    prev_low_1d = df_1d['low'].shift(1).values
    rang_1d = prev_high_1d - prev_low_1d
    R1_1d = prev_close_1d + 1.1 * rang_1d * 1.0 / 4
    S1_1d = prev_close_1d - 1.1 * rang_1d * 1.0 / 4
    
    # Align daily levels to 4h timeframe
    R1_1d_aligned = align_htf_to_ltf(prices, df_1d, R1_1d)
    S1_1d_aligned = align_htf_to_ltf(prices, df_1d, S1_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if (np.isnan(R1_1d_aligned[i]) or 
            np.isnan(S1_1d_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R1 + volume spike + price above daily EMA50 + ATR filter
            if (close[i] > R1_1d_aligned[i] and 
                volume_spike[i] and 
                close[i] > ema_50_1d_aligned[i] and
                close[i] > low[i] + 0.5 * atr[i]):  # price above low + half ATR
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 + volume spike + price below daily EMA50 + ATR filter
            elif (close[i] < S1_1d_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < ema_50_1d_aligned[i] and
                  close[i] < high[i] - 0.5 * atr[i]):  # price below high - half ATR
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters previous day's H-L range OR closes below daily EMA50
            if (close[i] < R1_1d_aligned[i] and close[i] > S1_1d_aligned[i]) or \
               close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters previous day's H-L range OR closes above daily EMA50
            if (close[i] < R1_1d_aligned[i] and close[i] > S1_1d_aligned[i]) or \
               close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-12 11:15
