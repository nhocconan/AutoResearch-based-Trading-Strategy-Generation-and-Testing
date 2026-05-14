# Strategy: 6h_Camarilla_R3S4_VolumeBreakout_Reversal_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.309 | +28.4% | -2.7% | 246 | PASS |
| ETHUSDT | 0.500 | +32.9% | -3.7% | 200 | PASS |
| SOLUSDT | -0.685 | +0.8% | -10.5% | 180 | FAIL |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.602 | -0.8% | -2.4% | 99 | FAIL |
| ETHUSDT | 0.479 | +9.3% | -3.5% | 88 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using daily Camarilla pivot levels with volume confirmation
# Camarilla levels (R3/S3 for reversals, R4/S4 for breakouts) provide institutional support/resistance
# Price breaking above R4 or below S4 with volume > 2x 20-period average indicates institutional breakout
# Price rejecting at R3 or S3 with volume confirmation indicates mean reversion opportunity
# Works in both bull/bear markets: breakouts capture trends, reversals capture pullbacks
# Target: 50-150 total trades over 4 years (12-37/year) with 0.25 position sizing

name = "6h_Camarilla_R3S4_VolumeBreakout_Reversal_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate daily Camarilla pivot levels ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla pivot levels
    # Pivot = (previous high + previous low + previous close) / 3
    pivot = (prev_high + prev_low + prev_close) / 3
    range_ = prev_high - prev_low
    
    # Resistance and Support levels
    r4 = pivot + (range_ * 1.1 / 2)
    r3 = pivot + (range_ * 1.1 / 4)
    s3 = pivot - (range_ * 1.1 / 4)
    s4 = pivot - (range_ * 1.1 / 2)
    
    # Align daily levels to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume confirmation: >2x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * vol_ma_20)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(s4_aligned[i]) or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price breaks above R4 with volume confirmation
            if close[i] > r4_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short breakout: price breaks below S4 with volume confirmation
            elif close[i] < s4_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
            # Long reversal: price rejects S3 with volume confirmation (bounce from support)
            elif close[i] < s3_aligned[i] and close[i] > s3_aligned[i] * 0.998 and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short reversal: price rejects R3 with volume correction (rejection from resistance)
            elif close[i] > r3_aligned[i] and close[i] < r3_aligned[i] * 1.002 and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below S3 (failed support) or reaches R4 (take profit)
            if close[i] < s3_aligned[i] or close[i] > r4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above R3 (failed resistance) or reaches S4 (take profit)
            if close[i] > r3_aligned[i] or close[i] < s4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-06 21:30
