# Strategy: 4h_Camarilla_R1S1_4hVolume_12hEMA50

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.350 | +42.9% | -15.7% | 133 | PASS |
| ETHUSDT | 0.199 | +32.1% | -13.9% | 131 | PASS |
| SOLUSDT | 1.112 | +262.0% | -26.9% | 138 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.786 | -4.7% | -9.9% | 54 | FAIL |
| ETHUSDT | 0.704 | +21.0% | -8.5% | 47 | PASS |
| SOLUSDT | 0.498 | +16.8% | -10.5% | 41 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R1/S1 breakout with 4h volume spike and 12h EMA trend filter.
# Long when price breaks above Camarilla R1 AND 4h volume > 1.5x 20-period average AND price > 12h EMA(50).
# Short when price breaks below Camarilla S1 AND 4h volume > 1.5x 20-period average AND price < 12h EMA(50).
# Exit when price crosses back below/above EMA(50) (trend-based exit).
# Uses Camarilla levels for precise reversal points, volume for confirmation, EMA for trend.
# Target: 100-200 total trades over 4 years (25-50/year) with controlled frequency.

name = "4h_Camarilla_R1S1_4hVolume_12hEMA50"
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
    
    # 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla levels using previous day's OHLC
    # Get daily data for prior day's OHLC
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla levels: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    camarilla_range = prev_high - prev_low
    r1 = prev_close + camarilla_range * 1.1 / 12
    s1 = prev_close - camarilla_range * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe (using previous day's levels)
    r1_4h = align_htf_to_ltf(prices, df_1d, r1)
    s1_4h = align_htf_to_ltf(prices, df_1d, s1)
    
    # 4h volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    # 12h EMA(50) for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for EMA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(r1_4h[i]) or np.isnan(s1_4h[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(ema_50_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above R1, volume spike, above 12h EMA50
            long_cond = (close[i] > r1_4h[i]) and volume_filter[i] and (close[i] > ema_50_12h_aligned[i])
            # Short conditions: price breaks below S1, volume spike, below 12h EMA50
            short_cond = (close[i] < s1_4h[i]) and volume_filter[i] and (close[i] < ema_50_12h_aligned[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below 12h EMA50 (trend change)
            if close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above 12h EMA50 (trend change)
            if close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-08 01:55
