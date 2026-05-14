# Strategy: 6h_camarilla_pivot_1d_trend_volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.547 | +47.6% | -7.6% | 203 | PASS |
| ETHUSDT | 0.012 | +19.5% | -12.4% | 188 | PASS |
| SOLUSDT | 0.828 | +119.2% | -17.1% | 168 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.742 | -1.7% | -6.6% | 81 | FAIL |
| ETHUSDT | 0.336 | +11.0% | -10.3% | 64 | PASS |
| SOLUSDT | 0.401 | +12.2% | -9.1% | 64 | PASS |

## Code
```python
#!/usr/bin/env python3
# 6h_camarilla_pivot_1d_trend_volume
# Hypothesis: Camarilla pivot levels from 1d combined with 1d EMA trend filter and volume confirmation.
# Long when price breaks above R4 with uptrend (price > 1d EMA50) and volume > 1.5x average.
# Short when price breaks below S4 with downtrend (price < 1d EMA50) and volume > 1.5x average.
# Fade trades at R3/S3 when price rejects these levels with confluence.
# Designed to capture strong breakouts and fade false moves in both bull and bear markets.
# Target: 50-150 total trades over 4 years (~12-37/year).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_camarilla_pivot_1d_trend_volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Camarilla levels from previous day
    # R4 = C + ((H-L) * 1.1/2)
    # R3 = C + ((H-L) * 1.1/4)
    # S3 = C - ((H-L) * 1.1/4)
    # S4 = C - ((H-L) * 1.1/2)
    # Where C, H, L are from previous day
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    # First day will have NaN due to roll, that's fine
    
    rang = prev_high - prev_low
    r4 = prev_close + (rang * 1.1 / 2)
    r3 = prev_close + (rang * 1.1 / 4)
    s3 = prev_close - (rang * 1.1 / 4)
    s4 = prev_close - (rang * 1.1 / 2)
    
    # Align Camarilla levels to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(avg_volume[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below S3 OR trend turns against us
            if (close[i] < s3_aligned[i]) or (close[i] < ema_50_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above R3 OR trend turns against us
            if (close[i] > r3_aligned[i]) or (close[i] > ema_50_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.5x average volume
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            
            # Breakout entries: R4 breakout (long) and S4 breakdown (short)
            if (close[i] > r4_aligned[i]) and (close[i] > ema_50_1d_aligned[i]) and volume_ok:
                position = 1
                signals[i] = 0.25
            elif (close[i] < s4_aligned[i]) and (close[i] < ema_50_1d_aligned[i]) and volume_ok:
                position = -1
                signals[i] = -0.25
            # Fade entries: Rejection at R3/S3 with trend confirmation
            elif (close[i] < r3_aligned[i] and close[i] > r3_aligned[i] * 0.995) and \
                 (close[i] > ema_50_1d_aligned[i]) and volume_ok:
                # Fade at R3 in uptrend - short
                position = -1
                signals[i] = -0.25
            elif (close[i] > s3_aligned[i] and close[i] < s3_aligned[i] * 1.005) and \
                 (close[i] < ema_50_1d_aligned[i]) and volume_ok:
                # Fade at S3 in downtrend - long
                position = 1
                signals[i] = 0.25
    
    return signals
```

## Last Updated
2026-04-08 13:22
