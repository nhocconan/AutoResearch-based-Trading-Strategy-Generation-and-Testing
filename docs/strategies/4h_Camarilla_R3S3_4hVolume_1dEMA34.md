# Strategy: 4h_Camarilla_R3S3_4hVolume_1dEMA34

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.419 | +48.3% | -14.6% | 87 | PASS |
| ETHUSDT | 0.149 | +27.5% | -16.3% | 92 | PASS |
| SOLUSDT | 0.908 | +183.3% | -27.3% | 90 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.364 | +0.6% | -6.7% | 38 | FAIL |
| ETHUSDT | 0.616 | +18.8% | -9.2% | 31 | PASS |
| SOLUSDT | 0.472 | +16.0% | -10.0% | 27 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 4h volume spike and 1d EMA34 trend filter.
# Long when price breaks above Camarilla R3 AND 4h volume > 1.8x 24-period average AND price > 1d EMA34.
# Short when price breaks below Camarilla S3 AND 4h volume > 1.8x 24-period average AND price < 1d EMA34.
# Exit when price crosses back below/above 1d EMA34 (trend-based exit).
# Uses tighter Camarilla levels (R3/S3) for stronger reversal points, higher volume threshold for confirmation.
# Target: 80-150 total trades over 4 years (20-37/year) with controlled frequency.

name = "4h_Camarilla_R3S3_4hVolume_1dEMA34"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate Camarilla levels using previous day's OHLC
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla levels: R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    camarilla_range = prev_high - prev_low
    r3 = prev_close + camarilla_range * 1.1 / 4
    s3 = prev_close - camarilla_range * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe (using previous day's levels)
    r3_4h = align_htf_to_ltf(prices, df_1d, r3)
    s3_4h = align_htf_to_ltf(prices, df_1d, s3)
    
    # 4h volume filter: current volume > 1.8x 24-period average
    vol_ma24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_filter = volume > (1.8 * vol_ma24)
    
    # 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Sufficient warmup for EMA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(r3_4h[i]) or np.isnan(s3_4h[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(ema34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above R3, volume spike, above 1d EMA34
            long_cond = (close[i] > r3_4h[i]) and volume_filter[i] and (close[i] > ema34_1d_aligned[i])
            # Short conditions: price breaks below S3, volume spike, below 1d EMA34
            short_cond = (close[i] < s3_4h[i]) and volume_filter[i] and (close[i] < ema34_1d_aligned[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below 1d EMA34 (trend change)
            if close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above 1d EMA34 (trend change)
            if close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-08 01:59
