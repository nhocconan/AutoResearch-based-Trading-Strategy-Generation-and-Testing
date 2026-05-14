# Strategy: 4h_Pivot_R1S1_Breakout_Volume_12h_Trend_Filter

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.057 | +19.0% | -13.2% | 632 | FAIL |
| ETHUSDT | 0.032 | +21.6% | -16.4% | 589 | PASS |
| SOLUSDT | -0.045 | +13.6% | -33.8% | 578 | FAIL |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.994 | +19.4% | -4.7% | 210 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
4h_Pivot_R1S1_Breakout_Volume_12h_Trend_Filter
Hypothesis: Trade Camarilla R1/S1 breakouts on 4h with volume confirmation, filtered by 12h trend direction (EMA50).
Long when price breaks above R1 with volume spike and 12h uptrend; short when breaks below S1 with volume spike and 12h downtrend.
Uses volume spike (volume > 1.5x 20-period average) to confirm breakout strength.
Target: 80-150 total trades over 4 years (20-38/year) with position size 0.25 to balance opportunity and risk.
Works in bull/bear: 12h trend filter avoids counter-trend trades, volume confirmation reduces false breakouts.
"""

name = "4h_Pivot_R1S1_Breakout_Volume_12h_Trend_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA50 for trend filter
    def ema(values, period):
        result = np.full_like(values, np.nan)
        if len(values) >= period:
            multiplier = 2.0 / (period + 1)
            result[period-1] = np.mean(values[:period])
            for i in range(period, len(values)):
                result[i] = multiplier * values[i] + (1 - multiplier) * result[i-1]
        return result
    
    ema50_12h = ema(close_12h, 50)
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate volume spike (volume > 1.5x 20-period average)
    vol_ma20 = np.full_like(volume, np.nan)
    for i in range(20, len(volume)):
        vol_ma20[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma20)
    
    # Calculate Camarilla levels from previous day (using 12h data as proxy for daily)
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), etc.
    # For intraday, we use previous period's range
    high_shift = np.roll(high, 1)
    low_shift = np.roll(low, 1)
    close_shift = np.roll(close, 1)
    high_shift[0] = high[0]
    low_shift[0] = low[0]
    close_shift[0] = close[0]
    
    # Previous period's range
    range_prev = high_shift - low_shift
    
    # Camarilla levels (using previous period's close as base)
    R1 = close_shift + 1.1 * range_prev / 12
    S1 = close_shift - 1.1 * range_prev / 12
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(R1[i]) or np.isnan(S1[i]) or 
            np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 with volume spike AND 12h uptrend (price > EMA50)
            if close[i] > R1[i] and volume_spike[i] and close[i] > ema50_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume spike AND 12h downtrend (price < EMA50)
            elif close[i] < S1[i] and volume_spike[i] and close[i] < ema50_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below S1 OR 12h trend turns down
            if close[i] < S1[i] or close[i] < ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above R1 OR 12h trend turns up
            if close[i] > R1[i] or close[i] > ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-20 04:22
