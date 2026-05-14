# Strategy: 4h_Camarilla_R4S4_4hVolume_1dEMA50

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.194 | +30.7% | -14.5% | 70 | PASS |
| ETHUSDT | -0.600 | -21.9% | -37.1% | 79 | FAIL |
| SOLUSDT | 0.811 | +152.5% | -30.9% | 59 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.006 | +5.6% | -6.7% | 24 | PASS |
| SOLUSDT | 0.139 | +7.3% | -15.5% | 22 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R4/S4 breakout with 4h volume spike and 1d EMA50 trend filter.
# Long when price breaks above Camarilla R4 AND 4h volume > 2.0x 24-period average AND price > 1d EMA50.
# Short when price breaks below Camarilla S4 AND 4h volume > 2.0x 24-period average AND price < 1d EMA50.
# Exit when price crosses back below/above 1d EMA50 (trend-based exit).
# Uses tighter R4/S4 levels (stronger reversal) and higher volume threshold (2.0x) to reduce trades.
# Target: 50-100 total trades over 4 years (12-25/year) for low fee drag.

name = "4h_Camarilla_R4S4_4hVolume_1dEMA50"
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
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla levels using previous day's OHLC
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla levels: R4 = C + (H-L)*1.1/2, S4 = C - (H-L)*1.1/2
    camarilla_range = prev_high - prev_low
    r4 = prev_close + camarilla_range * 1.1 / 2
    s4 = prev_close - camarilla_range * 1.1 / 2
    
    # Align Camarilla levels to 4h timeframe (using previous day's levels)
    r4_4h = align_htf_to_ltf(prices, df_1d, r4)
    s4_4h = align_htf_to_ltf(prices, df_1d, s4)
    
    # 4h volume filter: current volume > 2.0x 24-period average
    vol_ma24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_filter = volume > (2.0 * vol_ma24)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for EMA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(r4_4h[i]) or np.isnan(s4_4h[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(ema50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above R4, volume spike, above 1d EMA50
            long_cond = (close[i] > r4_4h[i]) and volume_filter[i] and (close[i] > ema50_1d_aligned[i])
            # Short conditions: price breaks below S4, volume spike, below 1d EMA50
            short_cond = (close[i] < s4_4h[i]) and volume_filter[i] and (close[i] < ema50_1d_aligned[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below 1d EMA50 (trend change)
            if close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above 1d EMA50 (trend change)
            if close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-08 02:00
