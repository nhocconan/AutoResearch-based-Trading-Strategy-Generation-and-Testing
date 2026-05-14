# Strategy: 6h_Camarilla_R3_S4_Breakout_1dTrend_VolumeSpike

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.789 | +49.6% | -4.7% | 151 | KEEP |
| ETHUSDT | 0.157 | +26.9% | -11.0% | 140 | KEEP |
| SOLUSDT | 0.465 | +54.8% | -15.6% | 121 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.071 | -2.0% | -5.2% | 63 | DISCARD |
| ETHUSDT | 1.786 | +29.9% | -5.2% | 51 | KEEP |
| SOLUSDT | -0.008 | +5.6% | -6.7% | 45 | DISCARD |

## Code
```python
#!/usr/bin/env python3
name = "6h_Camarilla_R3_S4_Breakout_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

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
    
    # === 1D DATA FOR CAMARILLA PIVOTS AND TREND ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate previous day's Camarilla levels
    # Using previous day's high, low, close to avoid look-ahead
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    prev_close_1d[0] = np.nan
    
    # Camarilla calculations
    range_1d = prev_high_1d - prev_low_1d
    camarilla_base = prev_close_1d
    
    # Resistance levels
    r3 = camarilla_base + range_1d * 1.1 / 4
    r4 = camarilla_base + range_1d * 1.1 / 2
    
    # Support levels
    s3 = camarilla_base - range_1d * 1.1 / 4
    s4 = camarilla_base - range_1d * 1.1 / 2
    
    # Align to 6h timeframe
    r3_6h = align_htf_to_ltf(prices, df_1d, r3)
    r4_6h = align_htf_to_ltf(prices, df_1d, r4)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3)
    s4_6h = align_htf_to_ltf(prices, df_1d, s4)
    
    # 1D EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_6h = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # === VOLUME SPIKE (20-period) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r3_6h[i]) or np.isnan(r4_6h[i]) or 
            np.isnan(s3_6h[i]) or np.isnan(s4_6h[i]) or
            np.isnan(ema34_1d_6h[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Break above R4 with volume spike + price above 1d EMA34 (uptrend)
            if (close[i] > r4_6h[i] and 
                close[i] > ema34_1d_6h[i] and
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Break below S4 with volume spike + price below 1d EMA34 (downtrend)
            elif (close[i] < s4_6h[i] and 
                  close[i] < ema34_1d_6h[i] and
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price breaks below R3 (false breakout) OR below EMA34
            if close[i] < r3_6h[i] or close[i] < ema34_1d_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above S3 (false breakout) OR above EMA34
            if close[i] > s3_6h[i] or close[i] > ema34_1d_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-12 05:54
