# Strategy: 4h_Camarilla_R1_S1_Breakout_12hEMA21_Trend_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.432 | +4.3% | -13.5% | 316 | FAIL |
| ETHUSDT | 0.141 | +26.7% | -10.4% | 281 | PASS |
| SOLUSDT | 0.633 | +73.3% | -23.8% | 238 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 1.021 | +20.1% | -5.4% | 107 | PASS |
| SOLUSDT | 0.166 | +7.8% | -11.8% | 90 | PASS |

## Code
```python
#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_12hEMA21_Trend_Volume
# Hypothesis: Combines Camarilla R1/S1 breakouts with 12h EMA21 trend filter and volume spike confirmation for high-probability institutional moves. Uses 12h EMA for multi-timeframe alignment to reduce false signals in both bull and bear markets. Target: 20-35 trades/year with strict entry conditions to minimize fee drag.

timeframe = "4h"
name = "4h_Camarilla_R1_S1_Breakout_12hEMA21_Trend_Volume"
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
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) == 0:
        return np.zeros(n)
    
    # Calculate EMA21 on 12h closes
    ema_21_12h = pd.Series(df_12h['close'].values).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_21_12h)
    
    # Get daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    # Calculate Camarilla levels on daily timeframe
    d_high = df_1d['high'].values
    d_low = df_1d['low'].values
    d_close = df_1d['close'].values
    
    camarilla_r1 = d_close + 1.1 * (d_high - d_low) / 12
    camarilla_s1 = d_close - 1.1 * (d_high - d_low) / 12
    
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume spike detection: 2x average volume (24-period = 1 day on 4h chart)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(24, 21)  # Ensure we have volume MA and EMA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema_21_12h_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: close > R1 with volume spike and 12h uptrend
            if close[i] > camarilla_r1_aligned[i] and volume[i] > 2.0 * vol_ma[i] and close[i] > ema_21_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: close < S1 with volume spike and 12h downtrend
            elif close[i] < camarilla_s1_aligned[i] and volume[i] > 2.0 * vol_ma[i] and close[i] < ema_21_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: touch S1 (opposite level) or trend failure
            if close[i] < camarilla_s1_aligned[i] or close[i] < ema_21_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: touch R1 (opposite level) or trend failure
            if close[i] > camarilla_r1_aligned[i] or close[i] > ema_21_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-07 01:49
