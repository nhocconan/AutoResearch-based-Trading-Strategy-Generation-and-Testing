# Strategy: 6H_Camarilla_R3_S3_Breakout_12hEMA50_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.284 | +29.8% | -5.3% | 296 | PASS |
| ETHUSDT | 0.023 | +21.7% | -8.3% | 267 | PASS |
| SOLUSDT | 0.220 | +32.3% | -10.5% | 232 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.959 | -6.5% | -7.5% | 124 | FAIL |
| ETHUSDT | 1.102 | +17.9% | -4.4% | 97 | PASS |
| SOLUSDT | 0.952 | +15.8% | -3.4% | 76 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla pivot breakout with 12h EMA50 trend filter and volume confirmation.
Long when price breaks above Camarilla R3 level (12h) AND price > 12h EMA50 (uptrend) AND volume > 1.8x average.
Short when price breaks below Camarilla S3 level (12h) AND price < 12h EMA50 (downtrend) AND volume > 1.8x average.
Exit when price reverts to Camarilla pivot point (PP) or trend reverses (price crosses 12h EMA50).
Uses 6h timeframe with tight entry conditions to avoid fee drag. Camarilla pivots from 12h provide intraday support/resistance.
12h EMA50 provides stable trend filter. Volume confirmation ensures high-conviction breakouts.
Target: 75-150 trades over 4 years (19-37/year) to stay within proven working range.
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
    
    # Load 6h data for Camarilla pivots - ONCE before loop
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 50:
        return np.zeros(n)
    
    close_6h = df_6h['close'].values
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    
    # Calculate Camarilla pivots for 6h
    PP_6h = (high_6h + low_6h + close_6h) / 3.0
    R1_6h = PP_6h + (high_6h - low_6h) * 1.1 / 12
    R2_6h = PP_6h + (high_6h - low_6h) * 1.1 / 6
    R3_6h = PP_6h + (high_6h - low_6h) * 1.1 / 4
    R4_6h = PP_6h + (high_6h - low_6h) * 1.1 / 2
    S1_6h = PP_6h - (high_6h - low_6h) * 1.1 / 12
    S2_6h = PP_6h - (high_6h - low_6h) * 1.1 / 6
    S3_6h = PP_6h - (high_6h - low_6h) * 1.1 / 4
    S4_6h = PP_6h - (high_6h - low_6h) * 1.1 / 2
    
    # Load 12h data for EMA50 trend filter - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 6h timeframe
    PP_6h_aligned = align_htf_to_ltf(prices, df_6h, PP_6h)
    R3_6h_aligned = align_htf_to_ltf(prices, df_6h, R3_6h)
    S3_6h_aligned = align_htf_to_ltf(prices, df_6h, S3_6h)
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume average (20-period) on 6h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(PP_6h_aligned[i]) or np.isnan(R3_6h_aligned[i]) or np.isnan(S3_6h_aligned[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        pp_val = PP_6h_aligned[i]
        r3_val = R3_6h_aligned[i]
        s3_val = S3_6h_aligned[i]
        ema50_val = ema50_12h_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_current = volume[i]
        price = close[i]
        
        if position == 0:
            # Long: price breaks above R3 AND price > 12h EMA50 (uptrend) AND volume confirmation
            if (price > r3_val and price > ema50_val and vol_current > 1.8 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 AND price < 12h EMA50 (downtrend) AND volume confirmation
            elif (price < s3_val and price < ema50_val and vol_current > 1.8 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price reverts to pivot point OR price breaks below 12h EMA50 (trend reversal)
                if price <= pp_val or price < ema50_val:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price reverts to pivot point OR price breaks above 12h EMA50 (trend reversal)
                if price >= pp_val or price > ema50_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Camarilla_R3_S3_Breakout_12hEMA50_Volume"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-23 01:39
