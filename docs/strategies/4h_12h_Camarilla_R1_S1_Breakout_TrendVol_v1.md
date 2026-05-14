# Strategy: 4h_12h_Camarilla_R1_S1_Breakout_TrendVol_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.117 | +25.0% | -10.9% | 250 | PASS |
| ETHUSDT | 0.442 | +40.0% | -9.9% | 239 | PASS |
| SOLUSDT | 0.472 | +53.4% | -22.7% | 202 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.321 | -4.1% | -6.9% | 92 | FAIL |
| ETHUSDT | 0.514 | +12.0% | -8.8% | 87 | PASS |
| SOLUSDT | 0.962 | +17.6% | -4.9% | 69 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
4h_12h_Camarilla_R1_S1_Breakout_TrendVol_v1
Hypothesis: 4-hour breakouts from daily Camarilla R1/S1 levels with 12-hour EMA50 trend filter and volume spike confirmation.
Only takes long when price breaks above R1 with volume spike and 12h uptrend, short when breaks below S1 with volume spike and 12h downtrend.
Uses tighter entry conditions (added ATR filter) to reduce trade frequency and improve robustness in both bull and bear markets.
"""

name = "4h_12h_Camarilla_R1_S1_Breakout_TrendVol_v1"
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
    
    # 12h data for Camarilla levels and trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # 12h Camarilla R1 and S1 from previous 12h period
    prev_close_12h = df_12h['close'].shift(1).values
    prev_high_12h = df_12h['high'].shift(1).values
    prev_low_12h = df_12h['low'].shift(1).values
    rang_12h = prev_high_12h - prev_low_12h
    R1_12h = prev_close_12h + 1.1 * rang_12h * 1.0 / 4
    S1_12h = prev_close_12h - 1.1 * rang_12h * 1.0 / 4
    
    # Align 12h levels to 4h timeframe
    R1_12h_aligned = align_htf_to_ltf(prices, df_12h, R1_12h)
    S1_12h_aligned = align_htf_to_ltf(prices, df_12h, S1_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if (np.isnan(R1_12h_aligned[i]) or 
            np.isnan(S1_12h_aligned[i]) or
            np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R1 + volume spike + price above 12h EMA50 + ATR filter
            if (close[i] > R1_12h_aligned[i] and 
                volume_spike[i] and 
                close[i] > ema_50_12h_aligned[i] and
                close[i] > low[i] + 0.5 * atr[i]):  # price above low + half ATR
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 + volume spike + price below 12h EMA50 + ATR filter
            elif (close[i] < S1_12h_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < ema_50_12h_aligned[i] and
                  close[i] < high[i] - 0.5 * atr[i]):  # price below high - half ATR
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters previous 12h period's H-L range OR closes below 12h EMA50
            if (close[i] < R1_12h_aligned[i] and close[i] > S1_12h_aligned[i]) or \
               close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters previous 12h period's H-L range OR closes above 12h EMA50
            if (close[i] < R1_12h_aligned[i] and close[i] > S1_12h_aligned[i]) or \
               close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-12 11:17
