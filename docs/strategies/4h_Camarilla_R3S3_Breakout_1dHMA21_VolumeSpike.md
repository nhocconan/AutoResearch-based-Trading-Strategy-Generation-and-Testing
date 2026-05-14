# Strategy: 4h_Camarilla_R3S3_Breakout_1dHMA21_VolumeSpike

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.021 | +21.2% | -8.0% | 277 | PASS |
| ETHUSDT | 0.268 | +34.2% | -13.0% | 255 | PASS |
| SOLUSDT | 0.587 | +73.1% | -27.6% | 210 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.566 | -7.8% | -10.1% | 103 | FAIL |
| ETHUSDT | 0.953 | +20.2% | -6.3% | 91 | PASS |
| SOLUSDT | -0.579 | -3.0% | -15.1% | 74 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 Breakout with 1d HMA21 Trend Filter and Volume Spike
# Long when price breaks above R3 (1d) AND price > 1d HMA21 (uptrend) AND volume spike
# Short when price breaks below S3 (1d) AND price < 1d HMA21 (downtrend) AND volume spike
# R3/S3 are stronger Camarilla levels (PP ± range/4) for fewer, higher-quality breaks
# HMA21 provides smooth trend filter with less lag than EMA/SMA
# Volume spike requires 2.0x 20-bar MA for confirmation
# Target: 75-200 total trades over 4 years (19-50/year) to avoid fee drag
# Works in bull (trend + breakouts) and bear (mean reversion at extremes + volume confirmation)
# Timeframe: 4h (primary timeframe as required)

name = "4h_Camarilla_R3S3_Breakout_1dHMA21_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Camarilla and HMA21
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d HMA21
    close_1d = df_1d['close'].values
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    wma_half = pd.Series(close_1d).ewm(span=half_len, adjust=False, min_periods=half_len).mean().values
    wma_full = pd.Series(close_1d).ewm(span=21, adjust=False, min_periods=21).mean().values
    wma_sqrt = pd.Series(2 * wma_half - wma_full).ewm(span=sqrt_len, adjust=False, min_periods=sqrt_len).mean().values
    hma_21_1d = wma_sqrt
    hma_21_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_21_1d)
    
    # Calculate Camarilla levels from previous 1d bar (HLC of completed daily bar)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_shifted = np.roll(close_1d, 1)
    high_1d_shifted = np.roll(high_1d, 1)
    low_1d_shifted = np.roll(low_1d, 1)
    
    # Calculate pivot point (PP) = (H+L+C)/3
    pp = (high_1d_shifted + low_1d_shifted + close_1d_shifted) / 3.0
    # Calculate range
    range_1d = high_1d_shifted - low_1d_shifted
    # Camarilla levels (R3/S3 = stronger levels at PP ± range/4)
    r3 = pp + (range_1d / 4.0)  # R3 = PP + range/4
    s3 = pp - (range_1d / 4.0)  # S3 = PP - range/4
    
    # Align Camarilla levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume confirmation on 4h
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume > (2.0 * vol_ma_20)  # Volume spike threshold
    else:
        volume_spike = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN (due to roll or insufficient data)
        if (np.isnan(hma_21_1d_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R3 AND uptrend (price > HMA21) AND volume spike
            if (close[i] > r3_aligned[i] and 
                close[i] > hma_21_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 AND downtrend (price < HMA21) AND volume spike
            elif (close[i] < s3_aligned[i] and 
                  close[i] < hma_21_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below R3 OR closes below HMA21
            if close[i] < r3_aligned[i] or close[i] < hma_21_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above S3 OR closes above HMA21
            if close[i] > s3_aligned[i] or close[i] > hma_21_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-05 12:52
