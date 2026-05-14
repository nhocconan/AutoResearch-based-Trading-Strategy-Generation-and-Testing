# Strategy: 6h_Camarilla_R3_S3_R4_S4_Breakout_MeanRev_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.121 | +15.9% | -13.6% | 188 | DISCARD |
| ETHUSDT | 0.534 | +48.0% | -9.9% | 168 | KEEP |
| SOLUSDT | 0.510 | +62.4% | -21.6% | 132 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 1.123 | +21.9% | -7.5% | 63 | KEEP |
| SOLUSDT | -0.336 | +1.2% | -10.1% | 52 | DISCARD |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot levels (HTF)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels from previous day
    # Pivot = (H + L + C) / 3
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Resistance and Support levels
    r1_1d = pivot_1d + (range_1d * 1.1 / 12)
    s1_1d = pivot_1d - (range_1d * 1.1 / 12)
    r2_1d = pivot_1d + (range_1d * 1.1 / 6)
    s2_1d = pivot_1d - (range_1d * 1.1 / 6)
    r3_1d = pivot_1d + (range_1d * 1.1 / 4)
    s3_1d = pivot_1d - (range_1d * 1.1 / 4)
    r4_1d = pivot_1d + (range_1d * 1.1 / 2)
    s4_1d = pivot_1d - (range_1d * 1.1 / 2)
    
    # Align Camarilla levels to 6h (properly delayed for completed 1d bar)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    r2_1d_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    s2_1d_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # Calculate 6h ATR (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 6h volume spike (volume > 2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # wait for volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(atr[i]) or
            np.isnan(r1_1d_aligned[i]) or
            np.isnan(s1_1d_aligned[i]) or
            np.isnan(r2_1d_aligned[i]) or
            np.isnan(s2_1d_aligned[i]) or
            np.isnan(r3_1d_aligned[i]) or
            np.isnan(s3_1d_aligned[i]) or
            np.isnan(r4_1d_aligned[i]) or
            np.isnan(s4_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirmed = volume_spike[i]
        
        if position == 0:
            # Long: price breaks above R4 with volume (strong bullish breakout)
            if close[i] > r4_1d_aligned[i-1] and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S4 with volume (strong bearish breakdown)
            elif close[i] < s4_1d_aligned[i-1] and vol_confirmed:
                signals[i] = -0.25
                position = -1
            # Long: mean reversion at S3 (price touches support and holds)
            elif close[i] <= s3_1d_aligned[i] and low[i] >= s3_1d_aligned[i] * 0.999 and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: mean reversion at R3 (price touches resistance and fails)
            elif close[i] >= r3_1d_aligned[i] and high[i] <= r3_1d_aligned[i] * 1.001 and vol_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below S1 or reverses at R3
            if close[i] < s1_1d_aligned[i] or (close[i] >= r3_1d_aligned[i] and high[i] <= r3_1d_aligned[i] * 1.001):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above R1 or reverses at S3
            if close[i] > r1_1d_aligned[i] or (close[i] <= s3_1d_aligned[i] and low[i] >= s3_1d_aligned[i] * 0.999):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R3_S3_R4_S4_Breakout_MeanRev_Volume"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-18 19:23
