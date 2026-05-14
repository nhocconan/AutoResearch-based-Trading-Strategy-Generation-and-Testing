# Strategy: 6h_Camarilla_S3_R3_Breakout_1d_VolumeSurge_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.090 | +24.1% | -12.6% | 218 | PASS |
| ETHUSDT | 0.535 | +54.0% | -9.2% | 191 | PASS |
| SOLUSDT | 0.962 | +132.8% | -15.8% | 162 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.746 | -11.2% | -15.4% | 87 | FAIL |
| ETHUSDT | 0.863 | +20.7% | -9.8% | 69 | PASS |
| SOLUSDT | -0.542 | -3.5% | -11.7% | 63 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Daily data for Camarilla pivot points
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla pivot points (S1, S2, S3, R1, R2, R3)
    range_1d = high_1d - low_1d
    close_prev = close_1d
    s1_1d = close_prev - (range_1d * 1.0 / 6)
    s2_1d = close_prev - (range_1d * 2.0 / 6)
    s3_1d = close_prev - (range_1d * 3.0 / 6)
    r1_1d = close_prev + (range_1d * 1.0 / 6)
    r2_1d = close_prev + (range_1d * 2.0 / 6)
    r3_1d = close_prev + (range_1d * 3.0 / 6)
    
    # Align daily Camarilla levels to 6h timeframe
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    s2_1d_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    r2_1d_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    
    # 6h ATR for volatility filter (14-period)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume filter (20-period MA)
    vol_ma20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_surge = prices['volume'].values > 2.0 * vol_ma20
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(200, n):
        # Skip if data not ready
        if (np.isnan(s1_1d_aligned[i]) or np.isnan(s2_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or
            np.isnan(r1_1d_aligned[i]) or np.isnan(r2_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or
            np.isnan(atr[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above S3 with volume surge (breakout of support)
            if close[i] > s3_1d_aligned[i] and vol_surge[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below R3 with volume surge (breakdown of resistance)
            elif close[i] < r3_1d_aligned[i] and vol_surge[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to S2 (for long) or R2 (for short)
            if position == 1:
                if close[i] < s2_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > r2_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_S3_R3_Breakout_1d_VolumeSurge_v1"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-22 05:32
