# Strategy: 12h_Camarilla_Pivot_R1S1_Breakout_1dTrend_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.135 | +16.8% | -5.1% | 103 | FAIL |
| ETHUSDT | 0.035 | +21.8% | -8.3% | 89 | PASS |
| SOLUSDT | -0.088 | +10.2% | -23.5% | 91 | FAIL |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.269 | +9.2% | -5.1% | 38 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
12h Camarilla Pivot R1/S1 Breakout with 1d Trend Filter and Volume Confirmation
Hypothesis: Camarilla pivot levels (R1/S1) act as key support/resistance. Breaking above R1 or below S1 with volume confirmation and 1d EMA trend filter captures institutional breakouts. Works in bull markets via upward breaks and in bear markets via downward breaks. Low trade frequency due to strict pivot-based entry.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels: R1, S1 based on previous day's range"""
    # Camarilla formula: R1 = close + (high - low) * 1.1/12
    #                  S1 = close - (high - low) * 1.1/12
    range_hl = high - low
    R1 = close + range_hl * 1.1 / 12
    S1 = close - range_hl * 1.1 / 12
    return R1, S1

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots and trend (once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivots on 1d (based on previous day's high/low/close)
    # We need to shift by 1 to use previous day's data for current day's levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivots using previous day's data
    R1_1d, S1_1d = calculate_camarilla(high_1d[:-1], low_1d[:-1], close_1d[:-1])
    # Prepend first value to maintain same length (no pivot for first day)
    R1_1d = np.concatenate([[np.nan], R1_1d])
    S1_1d = np.concatenate([[np.nan], S1_1d])
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d data to 12h timeframe
    R1_1d_aligned = align_htf_to_ltf(prices, df_1d, R1_1d)
    S1_1d_aligned = align_htf_to_ltf(prices, df_1d, S1_1d)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation: current volume > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Warmup for indicators
    
    for i in range(start_idx, n):
        if np.isnan(R1_1d_aligned[i]) or np.isnan(S1_1d_aligned[i]) or np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        r1_level = R1_1d_aligned[i]
        s1_level = S1_1d_aligned[i]
        trend = ema34_1d_aligned[i]
        vol_ok = vol_confirm[i]
        
        if position == 0:
            # Enter long: price breaks above R1 (resistance) with volume + uptrend
            if (not np.isnan(r1_level) and 
                vol_ok and 
                close[i] > r1_level and 
                close[i] > trend):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S1 (support) with volume + downtrend
            elif (not np.isnan(s1_level) and 
                  vol_ok and 
                  close[i] < s1_level and 
                  close[i] < trend):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below R1 or trend turns down
            if (not np.isnan(r1_level) and close[i] < r1_level) or close[i] < trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above S1 or trend turns up
            if (not np.isnan(s1_level) and close[i] > s1_level) or close[i] > trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_Pivot_R1S1_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0
```

## Last Updated
2026-04-18 06:33
