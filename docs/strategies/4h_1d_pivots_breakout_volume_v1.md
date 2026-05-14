# Strategy: 4h_1d_pivots_breakout_volume_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.758 | +55.5% | -9.7% | 121 | PASS |
| ETHUSDT | 0.048 | +22.1% | -10.2% | 107 | PASS |
| SOLUSDT | 0.449 | +57.1% | -17.4% | 112 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.644 | -0.1% | -7.0% | 49 | FAIL |
| ETHUSDT | 0.073 | +6.5% | -9.9% | 43 | PASS |
| SOLUSDT | 0.028 | +5.9% | -6.6% | 36 | PASS |

## Code
```python
#!/usr/bin/env python3
# 4h_1d_pivots_breakout_volume_v1
# Hypothesis: Trade breakouts of daily Camarilla pivot levels with volume confirmation on 4h timeframe.
# In bullish regime (price > 200-period EMA): long when price breaks above R4 pivot with volume surge.
# In bearish regime (price < 200-period EMA): short when price breaks below S4 pivot with volume surge.
# Uses volume filter (2x average) to confirm breakout strength.
# Camarilla levels provide institutional support/resistance levels that work in both trending and ranging markets.
# Designed for 4h timeframe to target 19-50 trades/year (75-200 total over 4 years).
# Works in both bull and bear markets by adapting to prevailing trend via EMA filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_pivots_breakout_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for previous day
    # Formula: R4 = C + ((H-L) * 1.1/2), R3 = C + ((H-L) * 1.1/4), etc.
    # We'll use the previous day's high, low, close to calculate today's levels
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    
    # First day will have NaN values due to roll
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Calculate Camarilla levels
    range_hl = prev_high - prev_low
    r4 = prev_close + (range_hl * 1.1 / 2)
    r3 = prev_close + (range_hl * 1.1 / 4)
    s3 = prev_close - (range_hl * 1.1 / 4)
    s4 = prev_close - (range_hl * 1.1 / 2)
    
    # Align pivot levels to 4h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # 200-period EMA for trend filter
    ema200 = pd.Series(close).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Volume confirmation: volume > 2x 6-period average (1 day of 4h bars)
    vol_ma_6 = pd.Series(volume).rolling(window=6, min_periods=6).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 200  # Ensure EMA is ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(ema200[i]) or np.isnan(vol_ma_6[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition
        vol_surge = volume[i] > 2.0 * vol_ma_6[i] if vol_ma_6[i] > 0 else False
        
        if position == 1:  # Long position
            # Exit: price breaks below S3 level (support)
            if close[i] < s3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above R3 level (resistance)
            if close[i] > r3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above R4 with volume surge and bullish trend
            if (close[i] > r4_aligned[i] and vol_surge and 
                close[i] > ema200[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below S4 with volume surge and bearish trend
            elif (close[i] < s4_aligned[i] and vol_surge and 
                  close[i] < ema200[i]):
                position = -1
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-08 18:34
