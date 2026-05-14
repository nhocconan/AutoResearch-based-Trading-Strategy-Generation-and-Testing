# Strategy: 6h_Camarilla_R3S3_Breakout_1wPivot_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.204 | +12.1% | -13.8% | 163 | FAIL |
| ETHUSDT | 0.033 | +21.3% | -17.7% | 135 | PASS |
| SOLUSDT | 0.450 | +55.1% | -25.0% | 109 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.045 | +6.3% | -7.9% | 41 | PASS |
| SOLUSDT | -0.408 | +1.3% | -11.3% | 32 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with weekly pivot filter and volume confirmation
# Uses daily Camarilla levels for entry/exit and weekly pivot for directional bias
# Weekly pivot acts as regime filter: price above weekly PP = long bias, below = short bias
# Volume spike (2.0x 24-bar MA) confirms institutional participation
# Designed for 50-150 total trades over 4 years (12-37/year) on 6h timeframe
# Works in bull markets (breakouts with trend) and bear markets (mean reversion at extremes)

name = "6h_Camarilla_R3S3_Breakout_1wPivot_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d OHLC for Camarilla levels (Pivot-based)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: R3, R4, S3, S4
    # Pivot = (High + Low + Close) / 3
    # Range = High - Low
    # R4 = Close + Range * 1.1/2
    # R3 = Close + Range * 1.1/4
    # S3 = Close - Range * 1.1/4
    # S4 = Close - Range * 1.1/2
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    r3_1d = close_1d + range_1d * 1.1 / 4.0
    r4_1d = close_1d + range_1d * 1.1 / 2.0
    s3_1d = close_1d - range_1d * 1.1 / 4.0
    s4_1d = close_1d - range_1d * 1.1 / 2.0
    
    # Align Camarilla levels to 6h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # Calculate 1w OHLC for weekly pivot (regime filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot point: (High + Low + Close) / 3
    weekly_pivot = (high_1w + low_1w + close_1w) / 3.0
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Volume confirmation: 2.0x 24-period average (24*6h = 6 days)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for volume MA)
    start_idx = 24
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Price breaks above R3 AND price > weekly pivot (bullish bias) AND volume spike
            if (close[i] > r3_1d_aligned[i] and 
                close[i] > weekly_pivot_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: Price breaks below S3 AND price < weekly pivot (bearish bias) AND volume spike
            elif (close[i] < s3_1d_aligned[i] and 
                  close[i] < weekly_pivot_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price breaks below S3 OR price below weekly pivot (regime change)
            if close[i] < s3_1d_aligned[i] or close[i] < weekly_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price breaks above R3 OR price above weekly pivot (regime change)
            if close[i] > r3_1d_aligned[i] or close[i] > weekly_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-02 08:07
