# Strategy: 4h_Camarilla_Pivot_R3S3_Breakout_1dEMA34_VolumeSpike

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.403 | +32.8% | -4.3% | 260 | PASS |
| ETHUSDT | 0.054 | +22.8% | -10.3% | 249 | PASS |
| SOLUSDT | 0.563 | +54.4% | -10.7% | 221 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -2.137 | -6.9% | -8.0% | 105 | FAIL |
| ETHUSDT | 0.110 | +7.0% | -8.5% | 89 | PASS |
| SOLUSDT | 0.628 | +11.4% | -4.2% | 73 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Camarilla_Pivot_R3S3_Breakout_1dEMA34_VolumeSpike"
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
    
    # Get 1d data for Camarilla pivot levels and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels (based on previous day's OHLC)
    # Using previous day's data for current day's levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point and Camarilla levels
    pivot = (high_1d + low_1d + close_1d) / 3
    range_hl = high_1d - low_1d
    
    # Camarilla levels: R3 = close + (high-low)*1.1/4, S3 = close - (high-low)*1.1/4
    r3 = close_1d + range_hl * 1.1 / 4
    s3 = close_1d - range_hl * 1.1 / 4
    
    # Calculate 34-period EMA on daily close for trend filter
    close_ser = pd.Series(close_1d)
    ema_34 = close_ser.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align all 1d indicators to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Get 1d volume average for spike detection
    vol_1d = pd.Series(df_1d['volume'].values)
    vol_ma20_1d = vol_1d.rolling(window=20, min_periods=20).mean().values
    vol_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20_1d)
    
    # Current 4h volume for confirmation (20-period MA)
    vol_series = pd.Series(volume)
    vol_ma20_current = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma20_1d_aligned[i]) or 
            np.isnan(vol_ma20_current[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 2.0 * vol_ma20_current[i]  # Volume spike filter
        
        if position == 0:
            # Long: Price breaks above R3 with volume spike and above EMA34
            if close[i] > r3_aligned[i] and vol_ok and close[i] > ema_34_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S3 with volume spike and below EMA34
            elif close[i] < s3_aligned[i] and vol_ok and close[i] < ema_34_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price falls back below R3
            if close[i] < r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price rises back above S3
            if close[i] > s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-09 10:30
