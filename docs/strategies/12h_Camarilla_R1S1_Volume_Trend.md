# Strategy: 12h_Camarilla_R1S1_Volume_Trend

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.039 | +18.1% | -18.1% | 187 | FAIL |
| ETHUSDT | 0.043 | +21.1% | -13.2% | 164 | PASS |
| SOLUSDT | 0.104 | +22.8% | -31.4% | 153 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.298 | +10.1% | -9.4% | 54 | PASS |
| SOLUSDT | -0.873 | -8.5% | -20.2% | 52 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with daily volume confirmation and Choppiness index regime filter.
# Long when price breaks above Camarilla R1 AND volume > 1.5x 20-period average AND Choppiness index < 61.8 (trending market).
# Short when price breaks below Camarilla S1 AND volume > 1.5x 20-period average AND Choppiness index < 61.8.
# Exit when price crosses back inside the Camarilla H-L range (between S1 and R1).
# Uses 12h timeframe as specified, with 1d Camarilla and volume for higher timeframe context.
# Target: 50-150 total trades over 4 years (12-37/year) with controlled frequency to avoid fee drag.

name = "12h_Camarilla_R1S1_Volume_Trend"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for Camarilla pivot calculation
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    prev_close = df_d['close'].shift(1).values
    prev_high = df_d['high'].shift(1).values
    prev_low = df_d['low'].shift(1).values
    prev_range = prev_high - prev_low
    
    # Camarilla R1 and S1 levels
    r1 = prev_close + (prev_range * 1.1 / 12)
    s1 = prev_close - (prev_range * 1.1 / 12)
    
    # Align Camarilla levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_d, s1)
    
    # Daily volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    # Choppiness index regime filter (14-period) on 12h data
    # Higher values indicate ranging market, lower values indicate trending
    atr_period = 14
    tr1 = np.maximum(high - low, np.absolute(high - np.roll(close, 1)))
    tr2 = np.maximum(np.absolute(low - np.roll(close, 1)), tr1)
    tr = np.where(np.arange(len(close)) == 0, high - low, tr2)  # First TR is just high-low
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Highest high and lowest low over the period
    highest_high = pd.Series(high).rolling(window=atr_period, min_periods=atr_period).max().values
    lowest_low = pd.Series(low).rolling(window=atr_period, min_periods=atr_period).min().values
    
    # Avoid division by zero
    range_hl = highest_high - lowest_low
    chop = np.where(range_hl != 0, 100 * np.log10(atr * atr_period / range_hl) / np.log10(atr_period), 50)
    
    # Trending market condition: Choppiness index < 61.8
    trending_filter = chop < 61.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, atr_period)  # Sufficient warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(trending_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R1, volume filter, trending market
            long_cond = (close[i] > r1_aligned[i]) and volume_filter[i] and trending_filter[i]
            # Short conditions: price breaks below Camarilla S1, volume filter, trending market
            short_cond = (close[i] < s1_aligned[i]) and volume_filter[i] and trending_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses back below Camarilla S1
            if close[i] < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses back above Camarilla R1
            if close[i] > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-08 01:21
